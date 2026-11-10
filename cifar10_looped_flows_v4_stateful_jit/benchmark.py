"""Real CIFAR-10 benchmark of ordered stateful flow sequences, in an isolated process."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from adam_atan2 import AdamATan2
from model import StatefulTRMJiT
from objective import sample_times, weighted_mse, interpolate
from data import load_data, normalize

p=argparse.ArgumentParser()
p.add_argument('--batch-size',type=int,required=True)
p.add_argument('--compile',action='store_true')
p.add_argument('--synthetic',action='store_true',help='Only for compile warmup before dataset is downloaded')
p.add_argument('--sequences',type=int,default=5)
a=p.parse_args()
torch.set_num_threads(3)
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32=True
net=StatefulTRMJiT().cuda()
ema=copy.deepcopy(net).requires_grad_(False)
opt=AdamATan2(net.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=1.)
if a.compile: net.core.compile(dynamic=False)
train = torch.randint(0,256,(2048,3,32,32),dtype=torch.uint8) if a.synthetic else load_data()[0]
report=dict(vars(a),gpu=torch.cuda.get_device_name(),torch=torch.__version__)
try:
    for seq in range(a.sequences):
        if seq==1:
            torch.cuda.synchronize(); started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        y=normalize(train[torch.randint(len(train),(a.batch_size,))])
        noise=torch.randn_like(y); times=sample_times(len(y),K=4)
        state=net.initial_state(len(y))
        for i in range(4):
            t=times[:,i]; x=interpolate(y,noise,t)
            opt.zero_grad(set_to_none=True)
            pred,state=net(x,t,state)
            loss=weighted_mse(pred,y,t)
            loss.backward(); opt.step()
            with torch.no_grad(): torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.001)
        print(json.dumps(dict(sequence=seq+1,loss=loss.item())),flush=True)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-started
    report.update(success=True,seconds_per_update=elapsed/((a.sequences-1)*4),
        images_per_second=a.batch_size*(a.sequences-1)*4/elapsed,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
except torch.OutOfMemoryError:
    report.update(success=False,error='CUDA OOM')
out=Path('runs/cifar10_looped_flows_v4_stateful_jit'); out.mkdir(parents=True,exist_ok=True)
(out/f'benchmark_{a.batch_size}_{"compile" if a.compile else "eager"}.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
