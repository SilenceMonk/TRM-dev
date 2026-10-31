"""Label-free generation; raw argmax, all 16 rounds, no solver/postprocessing."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from runtime import ROOT, atomic_json, build_model, setup_device, sha256
from data import valid_sudoku


@torch.inference_mode()
def evaluate_core(model,data_dir,split,batch_size,output,resume=False,limit=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    q=np.load(Path(data_dir)/f'q_{split}.npy',mmap_mode='r')
    s=np.load(Path(data_dir)/f's_{split}.npy',mmap_mode='r')
    n=len(q) if limit is None else min(limit,len(q))
    model.eval();start=0;totals=dict(exact=0,cells=0,blanks_correct=0,blanks=0,valid=0,clues=0)
    if resume and (output/'progress.json').exists():
        d=json.loads((output/'progress.json').read_text());start=d['processed'];totals=d['totals']
        assert d['n']==n and d['split']==split and d['batch_size']==batch_size
    predictions=output/'predictions.npy'
    pred=np.lib.format.open_memmap(predictions,mode='r+' if resume and predictions.exists() else 'w+',dtype=np.int8,shape=(n,81))
    t0=time.monotonic()
    for begin in range(start,n,batch_size):
        end=min(begin+batch_size,n);size=end-begin
        # Pad to fixed shape, so the last batch reuses the compiled graph.
        iq=np.zeros((batch_size,81),np.int32);iq[:size]=q[begin:end]+1
        batch={'inputs':torch.as_tensor(iq,device='cuda'), 'puzzle_identifiers':torch.zeros(batch_size,dtype=torch.int32,device='cuda')}
        with torch.device('cuda'):carry=model.initial_carry(batch)
        for _ in range(model.config.halt_max_steps):carry,out=model(carry=carry,batch=batch)
        digits=(out['logits'].argmax(-1)[:size]-1).cpu().numpy().astype(np.int8)
        pred[begin:end]=digits
        truth=s[begin:end];questions=q[begin:end];ok=digits==truth;blank=questions==0
        totals['exact']+=int(ok.all(-1).sum());totals['cells']+=int(ok.sum())
        totals['blanks_correct']+=int((ok&blank).sum());totals['blanks']+=int(blank.sum())
        totals['valid']+=int(valid_sudoku(digits).sum());totals['clues']+=int(((questions==0)|(digits==questions)).all(-1).sum())
        pred.flush()
        atomic_json(output/'progress.json',dict(processed=end,n=n,split=split,batch_size=batch_size,totals=totals))
        if begin==start or (begin//batch_size)%20==0:
            print(json.dumps(dict(evaluation=split,processed=end,n=n,exact_accuracy=totals['exact']/end,seconds=time.monotonic()-t0)),flush=True)
    result=dict(split=split,n=n,expected_n=len(q),complete=n==len(q),exact_accuracy=totals['exact']/n,
                cell_accuracy=totals['cells']/(81*n),blank_cell_accuracy=totals['blanks_correct']/max(1,totals['blanks']),
                sudoku_valid_rate=totals['valid']/n,clues_preserved_rate=totals['clues']/n,
                inference_rounds=16,raw_argmax=True,totals=totals)
    p_hat=result['exact_accuracy'];z=1.959963984540054
    center=(p_hat+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p_hat*(1-p_hat)/n+z*z/(4*n*n))/(1+z*z/n)
    result['exact_wilson_95']=[center-half,center+half]
    atomic_json(output/'metrics.json',result);print(json.dumps(result),flush=True)
    return result


def run_checkpoint(checkpoint,data_dir,split,batch_size,output,compile_mode='block',limit=None):
    setup_device();output=Path(output)
    signature=dict(checkpoint_sha=sha256(checkpoint),manifest_sha=sha256(Path(data_dir)/'manifest.json'),split=split,batch_size=batch_size,limit=limit)
    if (output/'signature.json').exists():assert json.loads((output/'signature.json').read_text())==signature,'Evaluation signature mismatch'
    else:atomic_json(output/'signature.json',signature)
    d=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=build_model(batch_size,compile_mode,loss=False);model.load_state_dict(d['model'],strict=True)
    return evaluate_core(model,data_dir,split,batch_size,output,resume=True,limit=limit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-dir',default=str(ROOT/'data'));p.add_argument('--split',choices=['train','val','test'],default='test')
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--output',required=True)
    p.add_argument('--compile',choices=['none','block','full'],default='block');p.add_argument('--limit',type=int)
    a=p.parse_args();run_checkpoint(a.checkpoint,a.data_dir,a.split,a.batch_size,a.output,a.compile,a.limit)
