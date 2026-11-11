"""Token-weighted, deterministic next-token cross entropy and perplexity."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
try:
    from .runtime import DATA, ModelConfig, atomic_json, sha256, setup
    from .model import TRMLanguageModel
    from .data import TokenDataset
except ImportError:
    from runtime import DATA, ModelConfig, atomic_json, sha256, setup
    from model import TRMLanguageModel
    from data import TokenDataset


@torch.no_grad()
def evaluate(model, data, batch_size=32, max_tokens=262144):
    was_training = model.training
    model.eval()
    count = data.blocks if max_tokens == 0 else min(data.blocks, max_tokens // data.seq_len)
    if count < 1:
        raise ValueError('Validation token budget must cover at least one block')
    # Fixed, evenly spaced windows across the whole official validation split.
    indices = np.linspace(0, data.blocks - 1, count, dtype=np.int64)
    total = torch.zeros((), device='cuda', dtype=torch.float64)
    tokens = 0
    for start in range(0, count, batch_size):
        x, y = data.batch(indices[start:start + batch_size])
        loss = model.loss(x, y)
        total += loss.double() * y.numel()
        tokens += y.numel()
    loss = total.item() / tokens
    model.train(was_training)
    if not math.isfinite(loss):
        raise FloatingPointError('Nonfinite validation loss')
    return dict(loss=loss, ppl=math.exp(loss), tokens=tokens, blocks=count,
                full_validation=count == data.blocks, split='official validation',
                token_coverage=tokens / (len(data.tokens) - 1),
                context=data.seq_len, protocol='nonoverlapping packed windows; state reset each window; EOS scored')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-dir', default=str(DATA))
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-tokens', type=int, default=0, help='0 evaluates every complete validation window')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--output')
    p.add_argument('--weights', choices=('auto', 'raw', 'ema'), default='auto')
    args = p.parse_args()
    setup()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ckpt['config']['data_manifest_sha256'] != sha256(Path(args.data_dir) / 'manifest.json'):
        raise ValueError('Checkpoint and evaluation data/tokenizer manifest differ')
    cfg = ModelConfig(**ckpt['model_config'])
    kind = ckpt.get('weights', 'raw') if args.weights == 'auto' else args.weights
    if kind == 'ema' and 'ema' in ckpt:
        state = ckpt['ema']
    elif kind == ckpt.get('weights', 'raw'):
        state = ckpt['model']
    else:
        raise ValueError(f'Checkpoint does not contain {kind} weights')
    model = TRMLanguageModel(cfg).cuda()
    model.load_state_dict(state)
    if args.compile:
        model.core.compile(dynamic=False)
    result = evaluate(model, TokenDataset(args.data_dir, 'validation', cfg.seq_len),
                      args.batch_size, args.max_tokens)
    result.update(step=ckpt['step'], checkpoint=str(Path(args.checkpoint).resolve()), weights=kind)
    print(json.dumps(result), flush=True)
    if args.output:
        atomic_json(args.output, result)


if __name__ == '__main__':
    main()
