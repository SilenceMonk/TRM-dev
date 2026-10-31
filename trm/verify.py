"""Checks which matter for a long scientific training run."""
import dataclasses
from pathlib import Path
import tempfile
import numpy as np
import torch
from runtime import Config, ROOT, atomic_json, sha256, model_config
from train import Trainer
from data import Sampler, valid_sudoku


def main():
    source=(ROOT/'trm/vendor/models/recursive_reasoning/trm.py').read_text()
    ours=(ROOT/'trm/model.py').read_text()
    assert ours.endswith(source)
    assert torch.cuda.is_available()
    cfg=Config(effective_batch=8,micro_batch=4,compile_mode='none')
    sampler=Sampler(cfg.data_dir,effective_batch=8,epochs_per_window=5)
    q,s=sampler.next()
    assert valid_sudoku(s-1).all()
    assert (((q-1)==0)|(q==s)).all()
    a=Trainer(cfg)
    values=a.update(q,s)
    assert torch.isfinite(values).all()
    with tempfile.TemporaryDirectory(prefix='trm-check-',dir=ROOT/'runs/trm') as tmp:
        ckpt=Path(tmp)/'resume.pt';a.save(ckpt,sampler)
        q2,s2=sampler.next()
        expected=a.update(q2,s2)
        expected_weights={k:v.clone() for k,v in a.model.state_dict().items()}
        b=Trainer(cfg)
        replay=Sampler(cfg.data_dir,effective_batch=8,epochs_per_window=5)
        b.load(ckpt,replay)
        qr,sr=replay.next()
        assert np.array_equal(qr,q2) and np.array_equal(sr,s2)
        actual=b.update(qr,sr)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        for k,v in b.model.state_dict().items():torch.testing.assert_close(v,expected_weights[k],atol=0,rtol=0)
    # Check microbatch accumulation against the same full effective batch.
    micro=Trainer(cfg);micro.update(q,s)
    full=Trainer(dataclasses.replace(cfg,micro_batch=8));full.update(q,s)
    maxdiff=max(float((p-dict(full.model.named_parameters())[k]).abs().max().detach()) for k,p in micro.model.named_parameters())
    assert maxdiff<2e-4, maxdiff
    # Check fused Adam-ATan2 against its source equation on random FP32 input.
    from adam_atan2 import AdamATan2
    torch.manual_seed(21)
    p=torch.nn.Parameter(torch.randn(1024,device='cuda'));ref=p.detach().clone()
    opt=AdamATan2([p],lr=1e-4,weight_decay=1,betas=(.9,.95))
    m=torch.zeros_like(p);v=torch.zeros_like(p)
    for step in range(1,4):
        g=torch.randn_like(p);p.grad=g;opt.step()
        m=.9*m+.1*g;v=.95*v+.05*g.square()
        ref=ref*(1-1e-4)-1e-4/(1-.9**step)*torch.atan2(m,v.sqrt()/(1-.95**step)**.5)
    torch.testing.assert_close(p,ref,atol=1e-6,rtol=1e-6)
    result=dict(official_model_source_exact=True,source_sha256=sha256(ROOT/'trm/vendor/models/recursive_reasoning/trm.py'),
        gpu_operation=True,finite_training=True,legal_training_augments=True,
        sampler_resume_exact=True,training_resume_bitwise_exact=True,
        fused_optimizer_equation=True,microbatch_vs_full_max_parameter_difference=maxdiff)
    atomic_json(ROOT/'runs/trm/verification.json',result)
    print(result,flush=True)

if __name__=='__main__':main()
