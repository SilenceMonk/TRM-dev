"""Synthetic steady-state local updates; isolates GPU cost from data downloads."""
import argparse
import json
import time
import torch
from adam_atan2 import AdamATan2
from model import LoopedFlow
p=argparse.ArgumentParser()
p.add_argument('--batch-size',type=int,default=128)
p.add_argument('--compile',action='store_true')
a=p.parse_args()
torch.set_num_threads(3)
torch.manual_seed(0)
net=LoopedFlow().cuda()
if a.compile: net.core.compile(dynamic=False)
opt=AdamATan2(net.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=1)
y=torch.randn(a.batch_size,3,32,32,device='cuda')
state=net.initial_state(len(y))
for i in range(12):
    if i==4:
        torch.cuda.synchronize(); start=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    t=torch.rand(len(y),device='cuda')
    opt.zero_grad(set_to_none=True)
    pred,state=net(y,t,state)
    loss=(pred-y).square().mean()
    loss.backward(); opt.step()
torch.cuda.synchronize()
print(json.dumps(dict(batch=a.batch_size,compile=a.compile,seconds_per_update=(time.monotonic()-start)/8,
                      peak_gib=torch.cuda.max_memory_allocated()/2**30,loss=loss.item())),flush=True)
