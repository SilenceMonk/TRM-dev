"""Fresh multi-epoch TRM training with raw/EMA full-validation selection and early stopping."""
import argparse
import copy
from dataclasses import asdict, dataclass
import fcntl
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
import torch
try:
    from .runtime import DATA, ROOT, ModelConfig, atomic_json, atomic_save, sha256, setup
    from .train import build, update
    from .data import TokenDataset
    from .evaluate import evaluate
except ImportError:
    from runtime import DATA, ROOT, ModelConfig, atomic_json, atomic_save, sha256, setup
    from train import build, update
    from data import TokenDataset
    from evaluate import evaluate


@torch.no_grad()
def update_ema(model, ema, decay):
    source = list(model.parameters())
    target = list(ema.parameters())
    torch._foreach_mul_(target, decay)
    torch._foreach_add_(target, source, alpha=1 - decay)
    # TRM buffers (h/l initial states and RoPE tables) are fixed after initialization.


@dataclass
class Plateau:
    lr: float
    patience: int = 3
    min_delta: float = .001
    min_epochs: int = 3
    factor: float = .5
    min_lr: float = 1e-6
    anchor: float = math.inf
    stale_epochs: int = 0
    stopped: bool = False

    def observe(self, score, epochs_completed):
        if not math.isfinite(score):
            raise FloatingPointError('Nonfinite epoch validation')
        improved = score < self.anchor - self.min_delta
        if improved:
            self.anchor = score
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
            self.lr = max(self.min_lr, self.lr * self.factor)
        self.stopped = epochs_completed >= self.min_epochs and self.stale_epochs >= self.patience
        return improved


def append_json(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')


def main(args):
    setup()
    if args.batch_size % args.micro_batch:
        raise ValueError('batch-size must be divisible by micro-batch')
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    lock = open(run / 'train.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = ModelConfig(seq_len=args.seq_len)
    manifest = json.loads((Path(args.data_dir) / 'manifest.json').read_text())
    assert cfg.vocab_size == manifest['vocab_size']
    train = TokenDataset(args.data_dir, 'train', cfg.seq_len)
    val = TokenDataset(args.data_dir, 'validation', cfg.seq_len)
    steps_per_epoch = math.ceil(train.blocks / args.batch_size)
    config = dict(model=asdict(cfg), seed=args.seed, initialization='random; no parent checkpoint',
        batch_size=args.batch_size, micro_batch=args.micro_batch, ema_decay=args.ema_decay,
        lr=args.lr, warmup=args.warmup, min_lr=args.min_lr, lr_factor=args.lr_factor,
        lr_schedule='linear initial warmup; constant until epoch plateau; then multiply by lr_factor',
        patience=args.patience, min_delta=args.min_delta, min_epochs=args.min_epochs,
        max_epochs=args.max_epochs, compile=args.compile, optimizer='AdamATan2 0.0.3 fused CUDA',
        betas=[.9, .95], weight_decay=1., grad_clip=1., Nsup=1,
        eval_every=args.eval_every, eval_tokens=args.eval_tokens, full_eval_every=args.full_eval_every,
        eval_batch=args.eval_batch, save_every=args.save_every, log_every=args.log_every,
        early_stopping='epoch-end min(raw full loss, EMA full loss); consecutive insignificant epochs',
        best_selection='lowest full validation loss separately for raw and EMA, including periodic full evaluations',
        data_manifest_sha256=sha256(Path(args.data_dir) / 'manifest.json'),
        train_blocks=train.blocks, steps_per_epoch=steps_per_epoch,
        code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    if (run / 'config.json').exists():
        if not args.resume:
            raise FileExistsError('Run exists; use --resume or a new directory')
        if json.loads((run / 'config.json').read_text()) != config:
            raise ValueError('Resume configuration/source mismatch')
    else:
        atomic_json(run / 'config.json', config)
    torch.manual_seed(args.seed)
    model, optimizer = build(cfg, compile_mode=False, lr=args.lr)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if args.compile:
        model.core.compile(dynamic=False)
        ema.core.compile(dynamic=False)
    atomic_json(run / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()),
        data_revision=manifest['revision']))
    step = cursor = epoch = ema_updates = 0
    elapsed_before = 0.
    sampler = np.random.default_rng(args.seed)
    order = sampler.permutation(train.blocks)
    plateau = Plateau(args.lr, args.patience, args.min_delta, args.min_epochs, args.lr_factor, args.min_lr)
    best_full = {}
    if args.resume and (run / 'latest.pt').exists():
        saved = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        step, cursor, epoch, ema_updates = (saved[k] for k in ('step', 'cursor', 'epoch', 'ema_updates'))
        order = saved['order']
        sampler.bit_generator.state = saved['sampler_rng']
        plateau = Plateau(**saved['plateau'])
        best_full = saved['best_full']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        elapsed_before = saved['elapsed_seconds']
        del saved
    # Preserve an already exported minimum if interrupted between export and latest save.
    for kind in ('raw', 'ema'):
        path = run / f'best_{kind}.json'
        if path.exists():
            result = json.loads(path.read_text())
            if kind not in best_full or result['loss'] < best_full[kind]['loss']:
                best_full[kind] = result
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    started = last_log = time.monotonic()
    last_step = step
    window_loss = torch.zeros((), device='cuda')
    window_count = 0

    def save():
        atomic_save(run / 'latest.pt', dict(schema=2, model_config=asdict(cfg), config=config,
            step=step, epoch=epoch, cursor=cursor, ema_updates=ema_updates, weights='raw',
            model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
            order=order, sampler_rng=sampler.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), plateau=asdict(plateau), best_full=best_full,
            elapsed_seconds=elapsed_before + time.monotonic() - started))

    def validate(full, reason):
        pair = {}
        for kind, network in (('raw', model), ('ema', ema)):
            start = time.monotonic()
            result = evaluate(network, val, args.eval_batch, 0 if full else args.eval_tokens)
            result.update(step=step, epoch=0 if step == 0 else epoch + 1, weights=kind,
                          reason=reason, seconds=time.monotonic() - start)
            pair[kind] = result
            append_json(run / 'validation.jsonl', result)
            print(json.dumps(dict(event='validation', **result)), flush=True)
            if full and (kind not in best_full or result['loss'] < best_full[kind]['loss']):
                export = dict(schema=2, model_config=asdict(cfg), config=config, step=step,
                    weights=kind, model=network.state_dict(), metrics=result)
                atomic_save(run / f'best_{kind}.pt', export)
                result = dict(result, checkpoint=f'best_{kind}.pt')
                best_full[kind] = result
                atomic_json(run / f'best_{kind}.json', result)
        atomic_json(run / ('validation_full_latest.json' if full else 'validation_subset_latest.json'), pair)
        atomic_json(run / 'validation_latest.json', pair)
        if best_full:
            atomic_json(run / 'best_overall.json', min(best_full.values(), key=lambda r: r['loss']))
        return pair

    if step == 0:
        validate(full=False, reason='random initialization baseline')
        save()
        last_log = time.monotonic()
    capped = bool(args.max_epochs and epoch >= args.max_epochs)
    while not plateau.stopped and not capped and not stop[0]:
        if cursor == len(order):
            cursor = 0
            order = sampler.permutation(train.blocks)
        chosen = order[cursor:cursor + args.batch_size]
        lr = plateau.lr * min(1., (step + 1) / args.warmup)
        for group in optimizer.param_groups:
            group['lr'] = lr
        batches = [train.batch(chosen[i:i + args.micro_batch]) for i in range(0, len(chosen), args.micro_batch)]
        loss, norm = update(model, optimizer, batches)
        update_ema(model, ema, args.ema_decay)
        step += 1
        ema_updates += 1
        cursor += len(chosen)
        window_loss += loss
        window_count += 1
        boundary = cursor == len(order)
        if step == 1 or step % args.log_every == 0:
            torch.cuda.synchronize()
            now = time.monotonic()
            seconds_per_step = (now - last_log) / (step - last_step)
            report = dict(step=step, epoch=epoch + 1, epochs_completed=epoch,
                epoch_fraction=cursor / len(order), phase='training', complete=False,
                train_loss=window_loss.item() / window_count, grad_norm=norm.item(), lr=lr,
                ema_decay=args.ema_decay, ema_updates=ema_updates,
                steps_per_epoch=steps_per_epoch, seconds_per_step=seconds_per_step,
                tokens_per_second=args.batch_size * cfg.seq_len / seconds_per_step,
                epoch_minutes_remaining=math.ceil((len(order) - cursor) / args.batch_size) * seconds_per_step / 60,
                elapsed_seconds=elapsed_before + now - started, stale_epochs=plateau.stale_epochs,
                best_full=best_full, peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                reserved_GiB=torch.cuda.max_memory_reserved() / 2**30)
            append_json(run / 'train.jsonl', report)
            atomic_json(run / 'status.json', report)
            print(json.dumps(report), flush=True)
            last_log, last_step = now, step
            window_loss.zero_()
            window_count = 0
        full_due = boundary or (args.full_eval_every and step % args.full_eval_every == 0)
        subset_due = step % args.eval_every == 0
        if (full_due or subset_due) and (not stop[0] or boundary):
            pair = validate(full=bool(full_due), reason='epoch end' if boundary else 'periodic')
            if boundary:
                epoch += 1
                improved = plateau.observe(min(r['loss'] for r in pair.values()), epoch)
                capped = bool(args.max_epochs and epoch >= args.max_epochs)
                summary = dict(epoch=epoch, step=step, validation=pair,
                    significant_improvement=improved, plateau=asdict(plateau), best_full=best_full)
                append_json(run / 'epochs.jsonl', summary)
                atomic_json(run / 'epoch_latest.json', summary)
                print(json.dumps(dict(event='epoch_complete', **summary)), flush=True)
            save()
            last_log, last_step = time.monotonic(), step
        elif step == 1 or step % args.save_every == 0 or stop[0]:
            save()
    save()
    reason = 'validation_plateau' if plateau.stopped else ('max_epochs' if capped else 'stopped_with_checkpoint')
    final = dict(step=step, epochs_completed=epoch, phase=reason, complete=plateau.stopped,
        plateau_reached=plateau.stopped, best_full=best_full, plateau=asdict(plateau),
        ema_updates=ema_updates, elapsed_seconds=elapsed_before + time.monotonic() - started)
    atomic_json(run / 'status.json', final)
    if plateau.stopped:
        atomic_json(run / 'RESULTS.json', final)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default=str(ROOT / 'runs/trm_tinystories/d256_seed0_ema'))
    p.add_argument('--data-dir', default=str(DATA))
    p.add_argument('--seq-len', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=224)
    p.add_argument('--micro-batch', type=int, default=224)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--ema-decay', type=float, default=.999)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--min-delta', type=float, default=.001)
    p.add_argument('--min-epochs', type=int, default=3)
    p.add_argument('--max-epochs', type=int, default=0, help='0 = no epoch cap, stop on validation plateau')
    p.add_argument('--lr-factor', type=float, default=.5)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--full-eval-every', type=int, default=1000)
    p.add_argument('--eval-tokens', type=int, default=262144)
    p.add_argument('--eval-batch', type=int, default=32)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.batch_size, args.micro_batch, args.seq_len, args.warmup, args.patience,
           args.min_epochs, args.eval_every, args.eval_batch, args.save_every, args.log_every) <= 0:
        raise ValueError('Positive training/evaluation sizes required')
    if min(args.max_epochs, args.full_eval_every, args.eval_tokens, args.min_delta) < 0:
        raise ValueError('Invalid stopping/evaluation budget')
    if not 0 < args.lr_factor < 1 or not 0 <= args.ema_decay < 1 or not 0 < args.min_lr <= args.lr:
        raise ValueError('Invalid learning-rate/EMA parameters')
    main(args)
