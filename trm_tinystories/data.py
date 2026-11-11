"""Pinned official train/validation splits; train-only byte-level BPE; uint16 mmap."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
try:
    from .runtime import DATA, atomic_json, sha256
except ImportError:
    from runtime import DATA, atomic_json, sha256

REPO = 'roneneldan/TinyStories'
REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'


def texts(paths):
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=['text']):
            yield from batch.column(0).to_pylist()


def prepare(out=DATA):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        print((out / 'manifest.json').read_text(), flush=True)
        return
    info = HfApi().dataset_info(REPO, revision=REVISION)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith('.parquet'))
    paths = {'train': [], 'validation': []}
    sources = []
    for name in files:
        print(f'Downloading {name}', flush=True)
        path = hf_hub_download(REPO, name, repo_type='dataset', revision=REVISION,
                               local_dir=out / 'raw')
        split = 'validation' if '/validation-' in name else 'train'
        paths[split].append(path)
        sources.append(dict(file=name, sha256=sha256(path)))
    tok_path = out / 'tokenizer.json'
    if not tok_path.exists():
        tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=8192, special_tokens=['<|endoftext|>', '<unk>'],
                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        def training_texts():
            for i, text in enumerate(texts(paths['train'])):
                if i >= 100000:
                    break
                yield text
        print('Training BPE on first 100000 training stories only', flush=True)
        tokenizer.train_from_iterator(training_texts(), trainer=trainer)
        tokenizer.save(str(tok_path))
    tokenizer = Tokenizer.from_file(str(tok_path))
    eos = tokenizer.token_to_id('<|endoftext|>')
    counts = {}
    for split, split_paths in paths.items():
        stories = tokens = 0
        target = out / f'{split}.bin'
        temp = target.with_suffix('.bin.tmp')
        with open(temp, 'wb') as f:
            batch = []
            def write_batch(batch):
                encoded = tokenizer.encode_batch(batch, add_special_tokens=False)
                values = [v for row in encoded for v in (*row.ids, eos)]
                np.asarray(values, dtype='<u2').tofile(f)
                return len(values)
            for text in texts(split_paths):
                batch.append(text)
                stories += 1
                if len(batch) == 2048:
                    tokens += write_batch(batch)
                    batch = []
                    if stories % 102400 == 0:
                        print(json.dumps(dict(split=split, stories=stories, tokens=tokens)), flush=True)
            if batch:
                tokens += write_batch(batch)
        os.replace(temp, target)
        counts[split] = dict(stories=stories, tokens=tokens, sha256=sha256(target))
    atomic_json(out / 'manifest.json', dict(repo=REPO, revision=REVISION, sources=sources,
        tokenizer_sha256=sha256(tok_path), vocab_size=tokenizer.get_vocab_size(),
        tokenizer_training='first 100000 official train stories; byte-level BPE 8192',
        dtype='little-endian uint16', eos_id=eos, splits=counts))
    print((out / 'manifest.json').read_text(), flush=True)


class TokenDataset:
    def __init__(self, directory, split, seq_len):
        self.tokens = np.memmap(Path(directory) / f'{split}.bin', mode='r', dtype='<u2')
        self.seq_len = seq_len
        self.blocks = (len(self.tokens) - 1) // seq_len

    def batch(self, indices, device='cuda'):
        import torch
        offsets = np.asarray(indices)[:, None] * self.seq_len + np.arange(self.seq_len + 1)
        array = np.asarray(self.tokens[offsets], dtype=np.int64)
        t = torch.from_numpy(array).pin_memory().to(device, non_blocking=True)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=str(DATA))
    prepare(p.parse_args().data_dir)
