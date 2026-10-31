"""Faithful TRM-MLP training with microbatching and complete resumable state."""
import argparse
import dataclasses
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from runtime import Config, ROOT, atomic_json, atomic_save, build_model, setup_device, sha256
from adam_atan2 import AdamATan2


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])


def cpu_tree(x):
    if isinstance(x,torch.Tensor): return x.detach().cpu()
    if dataclasses.is_dataclass(x): return {f.name:cpu_tree(getattr(x,f.name)) for f in dataclasses.fields(x)}
    if isinstance(x,dict): return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list): return [cpu_tree(v) for v in x]
    if isinstance(x,tuple): return tuple(cpu_tree(v) for v in x)
    return x


def restore_carry(d):
    from model import TinyRecursiveReasoningModel_ACTV1Carry as Carry, TinyRecursiveReasoningModel_ACTV1InnerCarry as Inner
    return Carry(Inner(**{k:v.cuda() for k,v in d['inner_carry'].items()}),
        d['steps'].cuda(),d['halted'].cuda(),{k:v.cuda() for k,v in d['current_data'].items()})


class Trainer:
    def __init__(self,cfg):
        self.cfg=cfg;setup_device()
        torch.manual_seed(cfg.seed);random.seed(cfg.seed);np.random.seed(cfg.seed)
        assert cfg.effective_batch%cfg.micro_batch==0
        self.model=build_model(cfg.micro_batch,cfg.compile_mode)
        self.emb=self.model.model.inner.puzzle_emb
        self.optimizer=AdamATan2(self.model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay,betas=(cfg.beta1,cfg.beta2))
        self.ema={k:p.detach().clone() for k,p in self.model.named_parameters() if p.requires_grad}
        self.carries=[None]*(cfg.effective_batch//cfg.micro_batch)
        self.step=0;self.best=-1.;self.elapsed=0.

    def update(self,q,s):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        lr=self.cfg.lr*min(1.,(self.step+1)/self.cfg.warmup)
        for g in self.optimizer.param_groups:g['lr']=lr
        emb_grad=torch.zeros_like(self.emb.weights)
        metrics_sum=None
        for i in range(len(self.carries)):
            a=i*self.cfg.micro_batch;b=a+self.cfg.micro_batch
            batch={'inputs':torch.as_tensor(q[a:b],device='cuda'),
                   'labels':torch.as_tensor(s[a:b],device='cuda'),
                   'puzzle_identifiers':torch.zeros(b-a,dtype=torch.int32,device='cuda')}
            if self.carries[i] is None:
                with torch.device('cuda'):self.carries[i]=self.model.initial_carry(batch)
            self.emb.local_weights.grad=None
            carry,loss,metrics,_,_=self.model(carry=self.carries[i],batch=batch,return_keys=[])
            (loss/self.cfg.effective_batch).backward()
            self.carries[i]=carry
            # Sudoku has a single shared puzzle identifier. Sum all microbatch
            # gradients BEFORE the official SignSGD update; never step per microbatch.
            emb_grad.add_(self.emb.local_weights.grad.sum(0,keepdim=True))
            packed=torch.stack([metrics[k].float() for k in ['count','accuracy','exact_accuracy','steps','lm_loss','q_halt_loss']])
            metrics_sum=packed if metrics_sum is None else metrics_sum+packed
        self.optimizer.step()
        with torch.no_grad():
            self.emb.weights.mul_(1-lr*self.cfg.weight_decay).add_(emb_grad.sign(),alpha=-lr)
            params=dict(self.model.named_parameters())
            for k,v in self.ema.items(): v.mul_(self.cfg.ema_rate).add_(params[k],alpha=1-self.cfg.ema_rate)
        self.emb.local_weights.grad=None
        self.step+=1
        return metrics_sum

    def save(self,path,sampler):
        atomic_save(path,dict(schema=1,config=dataclasses.asdict(self.cfg),step=self.step,best=self.best,
            model=cpu_tree(self.model.state_dict()),optimizer=cpu_tree(self.optimizer.state_dict()),
            ema=cpu_tree(self.ema),carries=cpu_tree(self.carries),rng=cpu_tree(rng_state()),
            sampler=sampler.state_dict(),elapsed=self.elapsed,
            manifest_sha=sha256(Path(self.cfg.data_dir)/'manifest.json')))

    def load(self,path,sampler):
        d=torch.load(path,map_location='cpu',weights_only=False)
        assert d['config']==dataclasses.asdict(self.cfg),'Resume configuration mismatch'
        assert d['manifest_sha']==sha256(Path(self.cfg.data_dir)/'manifest.json')
        self.model.load_state_dict(d['model']);self.optimizer.load_state_dict(d['optimizer'])
        # This custom fused optimizer keeps step counters on CUDA, unlike AdamW.
        for state in self.optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor): state[k]=v.cuda()
        self.ema={k:v.cuda() for k,v in d['ema'].items()};self.carries=[restore_carry(c) for c in d['carries']]
        self.step=d['step'];self.best=d['best'];self.elapsed=d['elapsed'];sampler.load_state_dict(d['sampler']);restore_rng(d['rng'])

    def export(self,path):
        state=cpu_tree(self.model.model.state_dict())
        for name,v in self.ema.items():state[name.removeprefix('model.')]=v.detach().cpu()
        atomic_save(path,dict(model=state,step=self.step,config=dataclasses.asdict(self.cfg),weights='ema',manifest_sha=sha256(Path(self.cfg.data_dir)/'manifest.json')))


def benchmark(args):
    cfg=Config(micro_batch=args.micro_batch,compile_mode=args.compile)
    trainer=Trainer(cfg)
    rng=np.random.default_rng(17)
    q=rng.integers(1,11,size=(cfg.effective_batch,81),dtype=np.int32)
    s=rng.integers(2,11,size=(cfg.effective_batch,81),dtype=np.int32)
    times=[]
    for i in range(args.benchmark):
        torch.cuda.synchronize();start=time.perf_counter()
        m=trainer.update(q,s)
        torch.cuda.synchronize();dt=time.perf_counter()-start;times.append(dt)
        print(json.dumps(dict(update=i+1,seconds=dt,peak_GiB=torch.cuda.max_memory_allocated()/2**30,finite=bool(torch.isfinite(m).all()))),flush=True)
    result=dict(micro_batch=cfg.micro_batch,compile_mode=cfg.compile_mode,seconds=times,
                median_steady_seconds=float(np.median(times[2:] or times)),peak_GiB=torch.cuda.max_memory_allocated()/2**30)
    if args.output:atomic_json(args.output,result)


def main(args):
    from data import Sampler
    run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=True)
    lock=open(run/'train.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=Config(micro_batch=args.micro_batch,compile_mode=args.compile)
    if args.steps is not None and not args.smoke:raise ValueError('--steps requires --smoke; full training budget cannot be shortened silently')
    if (run/'config.json').exists():
        assert json.loads((run/'config.json').read_text())==dataclasses.asdict(cfg)
        if not args.resume:raise FileExistsError('Run already exists; use --resume or a new directory')
    else:atomic_json(run/'config.json',dataclasses.asdict(cfg))
    code_manifest={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'trm').glob('*.py'))}
    if (run/'code_manifest.json').exists():
        assert json.loads((run/'code_manifest.json').read_text())==code_manifest, 'Training code changed since run creation'
    else:atomic_json(run/'code_manifest.json',code_manifest)
    atomic_json(run/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),source_sha=cfg.source_sha,data_sha=cfg.data_sha))
    trainer=Trainer(cfg);sampler=Sampler(cfg.data_dir,cfg.effective_batch,cfg.epochs_per_window)
    if args.resume and (run/'latest.pt').exists():trainer.load(run/'latest.pt',sampler)
    stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    target=args.steps if args.smoke else cfg.total_steps
    accumulated=None;t0=time.monotonic();previous=trainer.elapsed
    last_log=t0;last_step=trainer.step
    while trainer.step<target:
        q,s=sampler.next();m=trainer.update(q,s)
        accumulated=m if accumulated is None else accumulated+m
        trainer.elapsed=previous+time.monotonic()-t0
        if trainer.step%cfg.log_every==0 or trainer.step==1:
            values=accumulated.detach().cpu().tolist();count=max(values[0],1)
            report=dict(step=trainer.step,total_steps=target,phase='smoke' if args.smoke else 'training',
                complete=False,elapsed_seconds=trainer.elapsed,train_cell_accuracy=values[1]/count,
                train_exact_accuracy=values[2]/count,train_act_steps=values[3]/count,
                lm_loss_sum=values[4],q_halt_loss_sum=values[5],gpu_peak_GiB=torch.cuda.max_memory_allocated()/2**30)
            if not all(np.isfinite(values)):raise FloatingPointError('Nonfinite training metric')
            now=time.monotonic();report['seconds_per_update']=(now-last_log)/max(1,trainer.step-last_step)
            report['estimated_training_hours_remaining']=(target-trainer.step)*report['seconds_per_update']/3600
            last_log=now;last_step=trainer.step
            print(json.dumps(report),flush=True)
            with open(run/'train.jsonl','a') as f:f.write(json.dumps(report)+'\n')
            atomic_json(run/'status.json',report);accumulated=None
        save=trainer.step==1 or trainer.step%cfg.save_every==0 or stop[0] or trainer.step==target or trainer.step%cfg.val_every==0
        if save:
            trainer.save(run/'latest.pt',sampler)
            trainer.export(run/f'ema_step_{trainer.step:06d}.pt')
        if trainer.step%cfg.val_every==0 or trainer.step==target:
            from evaluate import evaluate_core
            saved=rng_state()
            current={k:p.detach().clone() for k,p in trainer.model.named_parameters()}
            with torch.no_grad():
                for k,p in trainer.model.named_parameters():p.copy_(trainer.ema[k])
            result=evaluate_core(trainer.model.model,cfg.data_dir,'val',cfg.micro_batch,run/f'val_step_{trainer.step:06d}',resume=True)
            with torch.no_grad():
                for k,p in trainer.model.named_parameters():p.copy_(current[k])
            restore_rng(saved)
            if result['exact_accuracy']>trainer.best:
                trainer.best=result['exact_accuracy'];trainer.export(run/f'best_ema_step_{trainer.step:06d}.pt')
                atomic_json(run/'best.json',dict(step=trainer.step,metrics=result,checkpoint=f'best_ema_step_{trainer.step:06d}.pt'))
            trainer.save(run/'latest.pt',sampler)
        if stop[0]:
            atomic_json(run/'status.json',dict(step=trainer.step,total_steps=target,complete=False,phase='stopped_with_checkpoint'))
            return
    atomic_json(run/'status.json',dict(step=trainer.step,total_steps=target,complete=False,phase='smoke_complete' if args.smoke else 'training_complete_pending_test',elapsed_seconds=trainer.elapsed))
    if not args.smoke:
        from evaluate import run_checkpoint
        final=run/f'ema_step_{trainer.step:06d}.pt'
        result=run_checkpoint(final,cfg.data_dir,'test',cfg.eval_batch,run/'official_test_final',cfg.compile_mode)
        atomic_json(run/'status.json',dict(step=trainer.step,total_steps=target,complete=result['complete'],phase='complete' if result['exact_accuracy']>=.85 else 'completed_below_reference_target',replication_target_met=result['exact_accuracy']>=.85,reference_exact_accuracy=.874,metrics=result,elapsed_seconds=trainer.elapsed))
        (run/'RESULTS.md').write_text(
            f"# TRM Sudoku-Extreme measured result\n\n"
            f"Optimizer updates: {trainer.step}/{target}\n\n"
            f"Official test: {result['n']}/{result['expected_n']} puzzles; complete={result['complete']}\n\n"
            f"Raw exact accuracy: {result['exact_accuracy']:.4%}; paper reference: 87.4%.\n\n"
            f"Reference tolerance met (>=85%): {result['exact_accuracy']>=.85}\n\n"
            f"Training elapsed (including validation/checkpointing): {trainer.elapsed/3600:.2f} hours.\n\n"
            "Protocol differences: fixed seed-0 1K subset (author indices unavailable); independent validation; microbatch accumulation; README 50K-epoch schedule.\n"
        )

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',default=str(ROOT/'runs/trm/sudoku_mlp_seed0'))
    p.add_argument('--micro-batch',type=int,default=256)
    p.add_argument('--compile',choices=['none','block','full'],default='full')
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--steps',type=int);p.add_argument('--benchmark',type=int,default=0);p.add_argument('--output')
    a=p.parse_args()
    if a.benchmark:benchmark(a)
    else:main(a)
