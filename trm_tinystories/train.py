"""Resumable width-256 causal TRM training, benchmarks, and validation tracking."""
import argparse
from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
import torch
from adam_atan2 import AdamATan2
try:
    from .runtime import DATA, ROOT, ModelConfig, atomic_json, atomic_save, sha256, setup
    from .model import TRMLanguageModel
    from .data import TokenDataset
    from .evaluate import evaluate
except ImportError:
    from runtime import DATA, ROOT, ModelConfig, atomic_json, atomic_save, sha256, setup
    from model import TRMLanguageModel
    from data import TokenDataset
    from evaluate import evaluate


def build(cfg, compile_mode=True, lr=3e-4):
    model = TRMLanguageModel(cfg).cuda()
    if compile_mode:
        model.core.compile(dynamic=False)
    optimizer = AdamATan2(model.parameters(), lr=lr, betas=(.9, .95), weight_decay=1.)
    return model, optimizer


def update(model, optimizer, batches):
    optimizer.zero_grad(set_to_none=True)
    losses = torch.zeros((), device='cuda')
    total_tokens = sum(y.numel() for _, y in batches)
    for x, y in batches:
        loss = model.loss(x, y)
        fraction = y.numel() / total_tokens
        (loss * fraction).backward()
        losses += loss.detach() * fraction
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return losses, norm


def benchmark(args):
    setup()
    torch.manual_seed(args.seed)
    cfg = ModelConfig(seq_len=args.seq_len)
    model, optimizer = build(cfg, args.compile)
    x = torch.randint(cfg.vocab_size, (args.micro_batch, cfg.seq_len + 1), device='cuda')
    records = []
    for i in range(args.benchmark):
        torch.cuda.synchronize()
        start = time.monotonic()
        loss, norm = update(model, optimizer, [(x[:, :-1].contiguous(), x[:, 1:].contiguous())])
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        row = dict(iteration=i, seconds=elapsed, tokens_per_second=args.micro_batch * cfg.seq_len / elapsed,
                   loss=loss.item(), grad_norm=norm.item(), peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                   reserved_GiB=torch.cuda.max_memory_reserved() / 2**30, micro_batch=args.micro_batch)
        records.append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        atomic_json(args.output, dict(model_config=asdict(cfg), records=records))


def main(args):
    setup()
    if args.batch_size % args.micro_batch:
        raise ValueError('batch-size must be divisible by micro-batch')
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    lock = open(run / 'train.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.manual_seed(args.seed)
    cfg = ModelConfig(seq_len=args.seq_len)
    manifest = json.loads((Path(args.data_dir) / 'manifest.json').read_text())
    assert cfg.vocab_size == manifest['vocab_size']
    train = TokenDataset(args.data_dir, 'train', cfg.seq_len)
    val = TokenDataset(args.data_dir, 'validation', cfg.seq_len)
    steps_per_epoch = math.ceil(train.blocks / args.batch_size)
    parent = None
    phase_start = 0
    if args.init_checkpoint:
        parent_path = Path(args.init_checkpoint).resolve()
        if parent_path.parent == run.resolve():
            raise ValueError('Continuation requires a new run directory')
        parent = torch.load(parent_path, map_location='cpu', weights_only=False)
        phase_start = parent['step']
    target = args.steps or phase_start + steps_per_epoch
    if target <= phase_start:
        raise ValueError('Target must be greater than the initialization step')
    code_hashes = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    config = dict(model=asdict(cfg), seed=args.seed, batch_size=args.batch_size, micro_batch=args.micro_batch,
                  steps=target, lr=args.lr, warmup=args.warmup, compile=args.compile,
                  optimizer='AdamATan2 0.0.3 fused CUDA', weight_decay=1., betas=[.9, .95],
                  lr_schedule='linear warmup then cosine to 10%', grad_clip=1., Nsup=1,
                  data_manifest_sha256=sha256(Path(args.data_dir) / 'manifest.json'),
                  code_sha256=code_hashes, eval_tokens=args.eval_tokens, eval_every=args.eval_every,
                  full_eval_every=args.full_eval_every,
                  train_blocks=train.blocks, steps_per_epoch=steps_per_epoch,
                  phase_start_step=phase_start)
    if parent is not None:
        for key in ('model', 'seed', 'batch_size', 'train_blocks', 'data_manifest_sha256',
                    'optimizer', 'weight_decay', 'betas', 'Nsup', 'grad_clip'):
            if parent['config'][key] != config[key]:
                raise ValueError(f'Initialization checkpoint mismatch: {key}')
        config['initialization'] = dict(checkpoint=str(parent_path),
            sha256=sha256(parent_path), step=phase_start, restore='model, optimizer, sampler, RNG')
    if (run / 'config.json').exists():
        if not args.resume:
            raise FileExistsError('Use --resume or a new run directory')
        if json.loads((run / 'config.json').read_text()) != config:
            raise ValueError('Resume configuration/source mismatch')
    else:
        atomic_json(run / 'config.json', config)
    model, optimizer = build(cfg, args.compile, args.lr)
    atomic_json(run / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()),
        data_revision=manifest['revision']))
    step = cursor = epoch = 0
    best = float('inf')
    elapsed_before = 0.
    sampler = np.random.default_rng(args.seed)
    order = sampler.permutation(train.blocks)
    resuming = args.resume and (run / 'latest.pt').exists()
    saved = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False) if resuming else parent
    if saved is not None:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        step, cursor, epoch, best = (saved[k] for k in ('step', 'cursor', 'epoch', 'best'))
        order = saved['order']
        sampler.bit_generator.state = saved['sampler_rng']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        elapsed_before = saved['elapsed_seconds']
        if not resuming:
            # Best selection starts with a measured baseline for this phase;
            # the parent's best checkpoint is preserved in its original run.
            best = float('inf')
    del parent, saved
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    started = time.monotonic()
    last_log = started
    last_step = step
    window_loss = torch.zeros((), device='cuda')
    window_count = 0

    def save(name='latest.pt'):
        atomic_save(run / name, dict(schema=1, model_config=asdict(cfg), config=config, step=step,
            model=model.state_dict(), optimizer=optimizer.state_dict(), cursor=cursor, epoch=epoch,
            order=order, sampler_rng=sampler.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), best=best,
            elapsed_seconds=elapsed_before + time.monotonic() - started))

    def validate(full=False):
        nonlocal best
        start = time.monotonic()
        result = evaluate(model, val, args.eval_batch, 0 if full else args.eval_tokens)
        result.update(step=step, weights='raw', seconds=time.monotonic() - start)
        with open(run / 'validation.jsonl', 'a') as f:
            f.write(json.dumps(result) + '\n')
        atomic_json(run / 'validation_latest.json', result)
        if full:
            atomic_json(run / 'validation_full_latest.json', result)
        print(json.dumps(dict(event='validation', **result)), flush=True)
        # Select best on a consistent subset; full final metrics are separate.
        if not full and result['loss'] < best:
            best = result['loss']
            save('best.pt')
            atomic_json(run / 'best.json', result)
        return result

    if step == phase_start:
        validate()
        if args.init_checkpoint:
            save()
    while step < target and not stop[0]:
        if cursor == len(order):
            epoch += 1
            cursor = 0
            order = sampler.permutation(train.blocks)
        chosen = order[cursor:cursor + args.batch_size]
        local_step = step - phase_start
        progress = max(0., (local_step - args.warmup) / max(1, target - phase_start - args.warmup))
        lr = args.lr * min(1., (local_step + 1) / args.warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        batches = [train.batch(chosen[i:i + args.micro_batch]) for i in range(0, len(chosen), args.micro_batch)]
        loss, norm = update(model, optimizer, batches)
        cursor += len(chosen)
        step += 1
        window_loss += loss
        window_count += 1
        if step == phase_start + 1 or step % args.log_every == 0:
            torch.cuda.synchronize()
            now = time.monotonic()
            seconds_per_step = (now - last_log) / (step - last_step)
            report = dict(step=step, total_steps=target, phase='training', complete=False,
                phase_step=step - phase_start, phase_total_steps=target - phase_start,
                train_loss=window_loss.item() / window_count, lr=lr, grad_norm=norm.item(),
                seconds_per_step=seconds_per_step, tokens_per_second=args.batch_size * cfg.seq_len / seconds_per_step,
                tokens_seen=(epoch * train.blocks + cursor) * cfg.seq_len,
                peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                elapsed_seconds=elapsed_before + now - started,
                training_hours_remaining=(target - step) * seconds_per_step / 3600)
            with open(run / 'train.jsonl', 'a') as f:
                f.write(json.dumps(report) + '\n')
            atomic_json(run / 'status.json', report)
            print(json.dumps(report), flush=True)
            last_log, last_step = now, step
            window_loss.zero_()
            window_count = 0
        if step == phase_start + 1 or step % args.save_every == 0 or stop[0] or step == target:
            save()
        if step % args.eval_every == 0 or step == target:
            validate()
            save()
            last_log, last_step = time.monotonic(), step
        if args.full_eval_every and step % args.full_eval_every == 0 and step < target:
            validate(full=True)
            last_log, last_step = time.monotonic(), step
    if stop[0]:
        save()
        atomic_json(run / 'status.json', dict(step=step, total_steps=target, phase='stopped_with_checkpoint', complete=False))
    else:
        final = validate(full=True)
        save()
        atomic_json(run / 'status.json', dict(step=step, total_steps=target, phase='complete', complete=True,
            full_validation=final, elapsed_seconds=elapsed_before + time.monotonic() - started))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default=str(ROOT / 'runs/trm_tinystories/d256_seed0'))
    p.add_argument('--data-dir', default=str(DATA))
    p.add_argument('--seq-len', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--micro-batch', type=int, default=128)
    p.add_argument('--steps', type=int, default=0, help='0 = one complete training epoch')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--init-checkpoint', help='Continue model/optimizer/sampler/RNG into a NEW run and LR phase')
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--full-eval-every', type=int, default=1000, help='0 disables periodic full validation')
    p.add_argument('--eval-tokens', type=int, default=262144)
    p.add_argument('--eval-batch', type=int, default=32)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--benchmark', type=int, default=0)
    p.add_argument('--output')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.batch_size, args.micro_batch, args.seq_len, args.warmup, args.eval_every,
           args.eval_batch, args.log_every, args.save_every) <= 0 or min(args.steps, args.eval_tokens, args.full_eval_every) < 0:
        raise ValueError('Invalid training/evaluation budget')
    if args.benchmark:
        benchmark(args)
    else:
        main(args)
