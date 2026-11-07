"""Official train split only; fixed 45k/5k seed-0 partition, normalized RGB."""
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10
ROOT = Path(__file__).resolve().parents[1]


def load_data():
    root = ROOT / 'data/cifar10'
    root.mkdir(parents=True, exist_ok=True)
    with open(root / 'download.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ds = CIFAR10(str(root), train=True, download=True)
    indices = np.random.default_rng(0).permutation(50000)
    split_path = root / 'split_seed0.npz'
    if split_path.exists():
        saved = np.load(split_path)
        assert np.array_equal(saved['train'], indices[:45000])
        assert np.array_equal(saved['val'], indices[45000:])
    else:
        np.savez(split_path, train=indices[:45000], val=indices[45000:])
    manifest = dict(train=45000, val=5000, split_seed=0,
                    dataset='torchvision.CIFAR10 official train (MD5 verified)',
                    split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                    raw_rgb_sha256=hashlib.sha256(ds.data.tobytes()).hexdigest(),
                    test_used=False, augmentation='optional horizontal flip, once per sequence')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    images = torch.from_numpy(ds.data.copy()).permute(0, 3, 1, 2).contiguous()
    return images[indices[:45000]], images[indices[45000:]]


def normalize(x, device='cuda'):
    return x.to(device=device, dtype=torch.float32).div_(127.5).sub_(1)


if __name__ == '__main__':
    a, b = load_data()
    print(a.shape, b.shape)
