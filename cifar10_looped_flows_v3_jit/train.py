"""Local optimizer steps, truncated recurrent state, EMA, atomic sequence-boundary resume."""
import argparse
import copy
import fcntl
import json
import hashlib
import os
from pathlib import Path
import signal
import time
import torch
from torchvision.utils import save_image
from adam_atan2 import AdamATan2
from model import TRMJiT, sample
from objective import weighted_mse, sample_times
from data import load_data, normalize, ROOT


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def atomic_save(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


@torch.no_grad()
def validate(model, data, batch=64, t_eps=0.05):
    gen = torch.Generator(device='cuda').manual_seed(2026)
    sums = torch.zeros(4, 5, device='cuda')
    for chunk in data.split(batch):
        y = normalize(chunk)
        noise = torch.randn(y.shape, generator=gen, device='cuda')
        for j, value in enumerate((.1, .3, .5, .7, .9)):
            t = torch.full((len(y),), value, device='cuda')
            x = (1-t[:, None, None, None])*noise + t[:, None, None, None]*y
            state = model.initial_state(len(y))
            for cycle in range(model.Nsup):
                pred, state = model.forward_cycle(x, t, state)
                sums[cycle, j] += (pred-y).square().mean((1, 2, 3)).sum()
    mse = (sums/len(data)).tolist()
    return dict(times=[.1,.3,.5,.7,.9], cycle_mse=mse, images=len(data),
                cycle_weighted_mse=[[v/max(1-t,t_eps)**2 for v,t in zip(row,[.1,.3,.5,.7,.9])] for row in mse])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='runs/cifar10_looped_flows_v3_jit/seed0')
    p.add_argument('--batch-size', type=int, default=768)
    p.add_argument('--steps', type=int, default=100000, help='Local optimizer updates, must be multiple of Nsup=4')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--val-images', type=int, default=5000)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--flip', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--p-mean', type=float, default=-0.8)
    p.add_argument('--p-std', type=float, default=0.8)
    p.add_argument('--t-eps', type=float, default=0.05)
    args = p.parse_args()
    assert args.p_std > 0 and args.t_eps > 0
    assert args.batch_size > 0 and args.steps > 0 and args.warmup > 0
    assert args.save_every > 0 and args.eval_every > 0 and 0 < args.val_images <= 5000
    assert args.steps % 4 == args.save_every % 4 == args.eval_every % 4 == 0
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    run = ROOT / args.run
    run.mkdir(parents=True, exist_ok=True)
    lock = open(run / 'train.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train, val = load_data()
    model = TRMJiT().cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = AdamATan2(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=1.)
    step = 0
    checkpoint = run / 'latest.pt'
    config = dict(vars(args), Nsup=4, H_cycles=3, L_cycles=6, optimizer='vendored AdamATan2 0.0.3 fused CUDA',
                  cuda_arch='sm'+''.join(map(str, torch.cuda.get_device_capability())),
                  betas=[.9,.95], weight_decay=1., ema=.999, lr_schedule='linear warmup then constant (original TRM)',
                  parameters=sum(p.numel() for p in model.parameters()),
                  output_head='JiT FinalLayer: affine RMSNorm eps1e-6, time shift/scale, zero-init biased linear; FP32',
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  time_sampling='one sigmoid(N(p_mean,p_std^2)) time per image; fixed across Nsup=4',
                  loss='mean(((prediction-target)/max(1-t,t_eps))^2)',
                  jit_source_sha='cbc743a2ada5e9762697da2c83f8c4f8379e8c17',
                  train_sampling='uniform with replacement', train_images=45000, val_images=args.val_images,
                  init='TRM truncated LeCun linear; fixed random h/l buffers',
                  source='trm/vendor @ c01103738605ba39d1430519b1ee0c62f4c707f8',
                  projection_scale=512**.5, integration='FP32 Euler; stateless BF16 attention TRM denoiser',
                  deep_supervision='4 local updates on same x,t,target; detached state between cycles',
                  architecture='2 shared attention blocks, D512, heads8, expansion4, RoPE, patch4',
                  position_encoding=dict(type='RoPE', theta=10000., head_dim=64, tokens=64))
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError('Run exists: use --resume or a new --run')
        d = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for key in ('parent_checkpoint', 'refine_start_step'):
            if key in d['config']:
                config[key] = d['config'][key]
        for key in ('batch_size', 'seed', 'lr', 'warmup', 'flip', 'steps', 'compile', 'p_mean', 'p_std', 't_eps', 'output_head', 'Nsup', 'H_cycles', 'L_cycles', 'architecture'):
            if d['config'][key] != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(d['model'])
        ema.load_state_dict(d['ema'])
        optimizer.load_state_dict(d['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.cuda()
        step = d['step']
        torch.set_rng_state(d['rng'])
        torch.cuda.set_rng_state_all(d['cuda_rng'])
    atomic_json(run / f'startup_step_{step:06d}.json', dict(
        start_step=step,
        loaded_checkpoint=str(checkpoint) if checkpoint.exists() else None,
        optimizer_state_entries=len(optimizer.state)))
    atomic_json(run / 'config.json', config)
    sources = list(Path(__file__).parent.glob('*.py')) + [ROOT/'trm/vendor/models/layers.py']
    atomic_json(run / f'source_manifest_{step:06d}.json',
                {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    if args.compile:
        model.core.compile(dynamic=False)
        ema.core.compile(dynamic=False)
    stop = False
    def terminate(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    fixed_noise = torch.randn(64, 3, 32, 32, device='cuda', generator=torch.Generator(device='cuda').manual_seed(1234))
    started = time.monotonic()
    start_step = step
    def save():
        atomic_save(checkpoint, dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                                   step=step, config=config, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
    if not checkpoint.exists():
        save()
    print(json.dumps(config), flush=True)
    with open(run / 'train.jsonl', 'a', buffering=1) as log:
        while step < args.steps and not stop:
            y = normalize(train[torch.randint(len(train), (args.batch_size,))])
            if args.flip:
                flip = torch.rand(len(y), device='cuda') < .5
                y = torch.where(flip[:,None,None,None], y.flip(-1), y)
            noise = torch.randn_like(y)
            t = sample_times(len(y), device='cuda', mean=args.p_mean, std=args.p_std)
            x = (1-t[:,None,None,None])*noise + t[:,None,None,None]*y
            state = model.initial_state(len(y))
            for i in range(4):
                optimizer.zero_grad(set_to_none=True)
                lr = args.lr*min(1., (step+1)/args.warmup)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                pred, state = model.forward_cycle(x, t, state)
                loss = weighted_mse(pred, y, t, args.t_eps)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at {step}')
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), .001)
                step += 1
                if step == 1 or step % 20 == 0:
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        image_emb, time_emb = model.embeddings(x, t)
                        rec = dict(step=step, sequences=(step+3)//4, supervision_cycle=i+1, loss=loss.item(), raw_mse=(pred-y).square().mean().item(),
                                   time_mean=t.mean().item(), time_min=t.min().item(), time_max=t.max().item(), lr=lr,
                                   seconds=time.monotonic()-started, seconds_per_update=(time.monotonic()-started)/(step-start_step),
                                   peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                                   image_embedding_rms=image_emb.float().square().mean().sqrt().item(),
                                   time_embedding_rms=time_emb.float().square().mean().sqrt().item())
                    log.write(json.dumps(rec)+'\n')
                    atomic_json(run / 'status.json', dict(rec, complete=False, pid=os.getpid()))
                    print(json.dumps(rec), flush=True)
            if step % args.save_every == 0 or stop or step == args.steps:
                save()
            if not stop and (step % args.eval_every == 0 or step == args.steps):
                metrics = validate(ema, val[:args.val_images], min(64,args.batch_size), args.t_eps)
                atomic_json(run / f'validation_{step:06d}.json', dict(step=step, **metrics))
                sample_metrics = {}
                for n in (8,16,32,64):
                    images = sample(ema, fixed_noise, n, t_eps=args.t_eps)
                    sample_metrics[n] = dict(mean=images.mean().item(), std=images.std().item(),
                                             out_of_range_fraction=(images.abs()>1).float().mean().item(),
                                             min=images.min().item(), max=images.max().item())
                    save_image(images.clamp(-1,1).add(1).div(2), run / f'samples_{step:06d}_N{n}.png', nrow=8)
                atomic_json(run / f'sample_stats_{step:06d}.json', sample_metrics)
                atomic_save(run / f'ema_{step:06d}.pt', dict(ema=ema.state_dict(), step=step, config=config))
        atomic_json(run / 'status.json', dict(step=step, complete=step==args.steps, phase='training_complete' if step==args.steps else 'stopped',
                                              fid_evaluated=False, peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__ == '__main__':
    main()
