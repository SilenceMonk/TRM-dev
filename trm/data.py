"""Immutable shared Sudoku split and fixed official-style augmentation pool."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from runtime import ROOT, DATA_SHA, SOURCE_SHA, atomic_json, sha256


def board(s):
    return np.frombuffer(s.replace('.', '0').encode(), dtype=np.uint8) - 48


def valid_sudoku(s):
    s = s.reshape(-1, 9, 9)
    target = np.arange(1, 10)
    return ((np.sort(s, axis=2) == target).all((1, 2)) &
            (np.sort(s, axis=1) == target[None, :, None]).all((1, 2)) &
            (np.sort(s.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9),axis=2)==target).all((1,2)))


def prepare(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if (path/'manifest.json').exists():
        m = json.loads((path/'manifest.json').read_text())
        assert m['data_sha'] == DATA_SHA and m['seed'] == 0
        for name, digest in m['files'].items():
            assert sha256(path/name) == digest, f'Changed shared data: {name}'
        print('Shared data verified', flush=True)
        return
    # Never overwrite an incomplete existing dataset; use a new directory.
    if list(path.iterdir()): raise FileExistsError(f'Nonempty data directory without manifest: {path}')
    sources = {s: hf_hub_download('sapientinc/sudoku-extreme',s+'.csv',repo_type='dataset',revision=DATA_SHA) for s in ['train','test']}
    with open(sources['train']) as f: n = sum(1 for _ in f) - 1
    rng = np.random.RandomState(0)
    train_ids = rng.choice(n,1000,replace=False)
    val_ids = rng.choice(np.setdiff1d(np.arange(n),train_ids),1000,replace=False)
    selected = {int(i):('train',j) for j,i in enumerate(train_ids)}
    selected.update({int(i):('val',j) for j,i in enumerate(val_ids)})
    arrays = {f'{k}_{s}':np.empty((1000,81),np.uint8) for s in ['train','val'] for k in ['q','s']}
    for split in ['train','test']:
        if split == 'test':
            with open(sources[split]) as f: nt = sum(1 for _ in f)-1
            arrays['q_test']=np.empty((nt,81),np.uint8)
            arrays['s_test']=np.empty((nt,81),np.uint8)
        with open(sources[split],newline='') as f:
            rows=csv.reader(f); next(rows)
            for i,(_,q,s,_) in enumerate(rows):
                if split == 'train' and i not in selected: continue
                dest,j=selected[i] if split=='train' else ('test',i)
                arrays[f'q_{dest}'][j]=board(q)
                arrays[f's_{dest}'][j]=board(s)
    assert len(arrays['q_test']) == 422786
    keys={s:{bytes(q) for q in arrays[f'q_{s}']} for s in ['train','val','test']}
    assert not (keys['train']&keys['val'] or keys['train']&keys['test'] or keys['val']&keys['test'])
    for split in ['train','val','test']:
        q,s=arrays[f'q_{split}'],arrays[f's_{split}']
        for i in range(0,len(q),8192):
            assert valid_sudoku(s[i:i+8192]).all()
            assert ((q[i:i+8192]==0)|(q[i:i+8192]==s[i:i+8192])).all()
        for k in ['q','s']: np.save(path/f'{k}_{split}.npy',arrays[f'{k}_{split}'])
    # Match upstream np.random.choice training subset, followed by its exact
    # shuffle_sudoku implementation; validation sampling uses a separate RNG above.
    import importlib.util
    # Extract function without importing upstream CLI/dependencies.
    source=(Path(__file__).parent/'vendor/dataset/build_sudoku_dataset.py').read_text()
    code=source[source.index('def shuffle_sudoku('):source.index('\ndef convert_subset(')]
    ns={'np':np}; exec(compile(code,'official_shuffle_sudoku','exec'),ns)
    np.random.seed(0); np.random.choice(n,1000,replace=False)
    pool_q=np.lib.format.open_memmap(path/'aug_q.npy', mode='w+',dtype=np.uint8,shape=(1000,1001,81))
    pool_s=np.lib.format.open_memmap(path/'aug_s.npy', mode='w+',dtype=np.uint8,shape=(1000,1001,81))
    for i in range(1000):
        q,s=arrays['q_train'][i].reshape(9,9),arrays['s_train'][i].reshape(9,9)
        pool_q[i,0],pool_s[i,0]=q.ravel(),s.ravel()
        for j in range(1,1001):
            aq,asol=ns['shuffle_sudoku'](q,s)
            pool_q[i,j],pool_s[i,j]=aq.ravel(),asol.ravel()
        if i%100==0: print(f'Augmenting {i}/1000',flush=True)
    pool_q.flush(); pool_s.flush()
    del pool_q,pool_s
    atomic_json(path/'manifest.json',dict(data_repo='sapientinc/sudoku-extreme', data_sha=DATA_SHA,
        source_sha=SOURCE_SHA,seed=0,n_train=1000,n_val=1000,n_test=422786,num_aug=1000,
        train_source_indices=train_ids.tolist(),val_source_indices=val_ids.tolist(),
        source_files={s:sha256(p) for s,p in sources.items()},
        files={p.name:sha256(p) for p in sorted(path.glob('*.npy'))},
        note='Official builder did not fix its seed or publish the selected 1K indices. Same protocol, not proven identical subset. Shared immutable split for TRM and future TRM-ODE.'))
    print('Shared dataset complete',flush=True)


class Sampler:
    """Replayable upstream grouped sampling, 5000 epochs per 6510-update window."""
    def __init__(self,data_dir,effective_batch=768,epochs_per_window=5000):
        self.q=np.load(Path(data_dir)/'aug_q.npy',mmap_mode='r')
        self.s=np.load(Path(data_dir)/'aug_s.npy',mmap_mode='r')
        self.batch=effective_batch; self.epochs=epochs_per_window
        self.window=0; self.position=0
        self.start_window()
    def start_window(self):
        self.rng=np.random.Generator(np.random.Philox(seed=self.window+1))
        self.order=np.concatenate([self.rng.permutation(1000) for _ in range(self.epochs)])
    def next(self):
        if self.position+self.batch>len(self.order):
            self.window+=1; self.position=0; self.start_window()
        groups=self.order[self.position:self.position+self.batch]
        # Scalar draws match the official _sample_batch loop.
        aug=np.array([self.rng.integers(0,1001) for _ in groups])
        self.position+=self.batch
        return self.q[groups,aug].astype(np.int32)+1,self.s[groups,aug].astype(np.int32)+1
    def state_dict(self):
        return dict(window=self.window,position=self.position,rng=self.rng.bit_generator.state)
    def load_state_dict(self,d):
        self.window=d['window'];self.position=d['position'];self.start_window()
        self.rng.bit_generator.state=d['rng']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default=str(ROOT/'data'))
    prepare(p.parse_args().data_dir)
