"""Compare one compiled/eager update using identical real Sudoku data and seeds."""
import argparse
import dataclasses
import gc
import numpy as np
import torch
from runtime import Config, ROOT, atomic_json
from train import Trainer
from data import Sampler

p=argparse.ArgumentParser();p.add_argument('--micro-batch',type=int,default=256);a=p.parse_args()
cfg=Config(micro_batch=a.micro_batch,compile_mode='full')
sampler=Sampler(cfg.data_dir);q,s=sampler.next()
t=Trainer(cfg);metrics=t.update(q,s).detach().cpu()
weights={k:v.detach().cpu().clone() for k,v in t.model.state_dict().items()}
del t;gc.collect();torch.cuda.empty_cache()
eager=Trainer(dataclasses.replace(cfg,compile_mode='none'))
ref=eager.update(q,s).detach().cpu()
differences={k:float((v.detach().cpu()-weights[k]).abs().max()) for k,v in eager.model.state_dict().items() if v.is_floating_point()}
maxdiff=max(differences.values())
# BF16 fusion changes rounding order. Bound actual parameter drift, not only loss.
assert maxdiff<2e-4,(maxdiff,sorted(differences.items(),key=lambda x:-x[1])[:5])
torch.testing.assert_close(metrics[-2:],ref[-2:],rtol=2e-3,atol=2e-3)
result=dict(micro_batch=a.micro_batch,compiled_eager_max_parameter_difference=maxdiff,compiled_metrics=metrics.tolist(),eager_metrics=ref.tolist(),passed=True)
atomic_json(ROOT/f'runs/trm/compilation_check_b{a.micro_batch}.json',result)
print(result,flush=True)
