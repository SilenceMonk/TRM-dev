"""Local optimizer steps, truncated recurrent state, EMA, atomic sequence-boundary resume."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import torch
from torchvision.utils import save_image
from adam_atan2 import AdamATan2
from model import LoopedFlow, sample
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
def validate(model, data, batch=64):
    gen = torch.Generator(device='cuda').manual_seed(2026)
    sums = torch.zeros(2, 5, device='cuda')
    for chunk in data.split(batch):
        y = normalize(chunk)
        noise = torch.randn(y.shape, generator=gen, device='cuda')
        state = model.initial_state(len(y))
        for j in range(5):
            t = torch.full((len(y),), (j+.5)/5, device='cuda')
            x = (1-t[:, None, None, None])*noise + t[:, None, None, None]*y
            pred, state = model(x, t, state)
            reset, _ = model(x, t, model.initial_state(len(y)))
            sums[0, j] += (pred-y).square().mean((1, 2, 3)).sum()
            sums[1, j] += (reset-y).square().mean((1, 2, 3)).sum()
    return dict(times=[.1,.3,.5,.7,.9], carried_mse=(sums[0]/len(data)).tolist(),
                reset_mse=(sums[1]/len(data)).tolist(), images=len(data))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='runs/cifar10_looped_flows/seed0_conv_scratch')
    p.add_argument('--batch-size', type=int, default=640)
    p.add_argument('--steps', type=int, default=100000, help='Local optimizer updates, must be multiple of K=4')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--val-images', type=int, default=5000)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--flip', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
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
    model = LoopedFlow().cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = AdamATan2(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=1.)
    step = 0
    checkpoint = run / 'latest.pt'
    config = dict(vars(args), K=4, R=3, m=6, optimizer='vendored AdamATan2 0.0.3 fused CUDA sm120',
                  betas=[.9,.95], weight_decay=1., ema=.999, lr_schedule='linear warmup then constant (original TRM)',
                  parameters=sum(p.numel() for p in model.parameters()),
                  output_head='unpatchify + zero-init residual Conv2d(3,3,3,padding=1)',
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  time_sampling='sort K+1 independent uniforms, use first K',
                  train_sampling='uniform with replacement', train_images=45000, val_images=args.val_images,
                  init='TRM truncated LeCun linear; fixed random h/l buffers',
                  source='trm/vendor @ c01103738605ba39d1430519b1ee0c62f4c707f8',
                  projection_scale=512**.5, integration='FP32 Euler; BF16 denoiser')
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError('Run exists: use --resume or a new --run')
        d = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for key in ('parent_checkpoint', 'refine_start_step'):
            if key in d['config']:
                config[key] = d['config'][key]
        for key in ('batch_size', 'seed', 'lr', 'warmup', 'flip', 'steps', 'compile'):
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
        optimizer_state_entries=len(optimizer.state),
        refine_weight_nonzero=model.refine.weight.count_nonzero().item(),
        refine_bias_nonzero=model.refine.bias.count_nonzero().item()))
    atomic_json(run / 'config.json', config)
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
    print(json.dumps(config), flush=True)
    with open(run / 'train.jsonl', 'a', buffering=1) as log:
        while step < args.steps and not stop:
            y = normalize(train[torch.randint(len(train), (args.batch_size,))])
            if args.flip:
                flip = torch.rand(len(y), device='cuda') < .5
                y = torch.where(flip[:,None,None,None], y.flip(-1), y)
            noise = torch.randn_like(y)
            times = torch.rand(len(y), 5, device='cuda').sort(dim=1).values[:, :4]
            state = model.initial_state(len(y))
            for i in range(4):
                t = times[:, i]
                x = (1-t[:,None,None,None])*noise + t[:,None,None,None]*y
                optimizer.zero_grad(set_to_none=True)
                lr = args.lr*min(1., (step+1)/args.warmup)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                pred, state = model(x, t, state)
                loss = (pred-y).square().mean()
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
                        rec = dict(step=step, sequences=step//4, loss=loss.item(), lr=lr,
                                   seconds=time.monotonic()-started, seconds_per_update=(time.monotonic()-started)/(step-start_step),
                                   peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                                   image_embedding_rms=image_emb.float().square().mean().sqrt().item(),
                                   time_embedding_rms=time_emb.float().square().mean().sqrt().item())
                    log.write(json.dumps(rec)+'\n')
                    atomic_json(run / 'status.json', dict(rec, complete=False, pid=os.getpid()))
                    print(json.dumps(rec), flush=True)
            if step % args.save_every == 0 or stop or step == args.steps:
                save()
            if step % args.eval_every == 0 or step == args.steps:
                metrics = validate(ema, val[:args.val_images], min(64,args.batch_size))
                atomic_json(run / f'validation_{step:06d}.json', dict(step=step, **metrics))
                for n in (8,16,32,64):
                    images = sample(ema, fixed_noise, n)
                    save_image(images.clamp(-1,1).add(1).div(2), run / f'samples_{step:06d}_N{n}.png', nrow=8)
                atomic_save(run / f'ema_{step:06d}.pt', dict(ema=ema.state_dict(), step=step, config=config))
        atomic_json(run / 'status.json', dict(step=step, complete=step==args.steps, phase='training_complete' if step==args.steps else 'stopped',
                                              fid_evaluated=False, peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__ == '__main__':
    main()
