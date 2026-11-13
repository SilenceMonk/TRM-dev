"""Raw/EMA evaluation of label-free per-token ACT; fixed-16 mode is a diagnostic."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
try:
    from .runtime import DATA, atomic_json, sha256, setup
    from .model_act import ACTConfig, TokenACT
    from .data import TokenDataset
except ImportError:
    from runtime import DATA, atomic_json, sha256, setup
    from model_act import ACTConfig, TokenACT
    from data import TokenDataset


@torch.no_grad()
def evaluate(model, data, batch_size=32, max_tokens=262144, force_max=False):
    was_training = model.training
    model.eval()
    count = data.blocks if max_tokens == 0 else min(data.blocks, max_tokens // data.seq_len)
    if count < 1:
        raise ValueError('Validation budget must cover a complete window')
    indices = np.linspace(0, data.blocks - 1, count, dtype=np.int64)
    total = torch.zeros((), device='cuda', dtype=torch.float64)
    histogram = torch.zeros(model.config.halt_max_steps, device='cuda', dtype=torch.int64)
    tokens = 0
    for start in range(0, count, batch_size):
        x, y = data.batch(indices[start:start + batch_size])
        logits, steps = model.forward_with_steps(x, force_max=force_max)
        total += F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), reduction='sum').double()
        histogram += torch.bincount(steps.flatten().long() - 1, minlength=len(histogram))
        tokens += y.numel()
    loss = total.item() / tokens
    hist = histogram.tolist()
    model.train(was_training)
    if not math.isfinite(loss):
        raise FloatingPointError('Nonfinite ACT validation')
    return dict(loss=loss, ppl=math.exp(loss), tokens=tokens, blocks=count,
        full_validation=count == data.blocks, split='official validation', context=data.seq_len,
        token_coverage=tokens / (len(data.tokens) - 1), halt_histogram=hist,
        mean_cycles=sum((i + 1) * n for i, n in enumerate(hist)) / tokens,
        fraction_at_cap=hist[-1] / tokens, halt_max_steps=model.config.halt_max_steps,
        inference='fixed_max' if force_max else 'token_act',
        protocol='packed nonoverlapping windows; no labels in halting; frozen halted states; EOS scored')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--weights', choices=('auto', 'raw', 'ema'), default='auto')
    p.add_argument('--data-dir', default=str(DATA))
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-tokens', type=int, default=0)
    p.add_argument('--force-max', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--output')
    args = p.parse_args()
    setup()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ckpt['config']['data_manifest_sha256'] != sha256(Path(args.data_dir) / 'manifest.json'):
        raise ValueError('Data/tokenizer manifest mismatch')
    kind = ckpt.get('weights', 'raw') if args.weights == 'auto' else args.weights
    if kind == 'ema' and 'ema' in ckpt:
        state = ckpt['ema']
    elif kind == ckpt.get('weights', 'raw'):
        state = ckpt['model']
    else:
        raise ValueError(f'Checkpoint does not contain {kind} weights')
    cfg = ACTConfig(**ckpt['model_config'])
    model = TokenACT(cfg).cuda()
    model.load_state_dict(state)
    if args.compile:
        model.core.compile(dynamic=False)
    result = evaluate(model, TokenDataset(args.data_dir, 'validation', cfg.seq_len),
                      args.batch_size, args.max_tokens, args.force_max)
    result.update(weights=kind, step=ckpt['step'])
    print(json.dumps(result), flush=True)
    if args.output:
        atomic_json(args.output, result)


if __name__ == '__main__':
    main()
