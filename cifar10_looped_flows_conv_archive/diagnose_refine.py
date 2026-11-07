"""Small-batch diagnosis; does not change the training model or checkpoints."""
import argparse
import json
from pathlib import Path
import torch
from model import LoopedFlow, patchify
from data import ROOT

parser=argparse.ArgumentParser()
parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
args=parser.parse_args()
device=args.device
torch.set_num_threads(2)
base=ROOT/'runs/cifar10_looped_flows'
old=torch.load(base/'seed0/latest.pt',map_location='cpu',weights_only=False)
new=torch.load(base/'seed0_refine_from29000/ema_045000.pt',map_location='cpu',weights_only=False)
net=LoopedFlow().to(device).eval().requires_grad_(False)
if device == 'cpu':
    # FP32 CPU diagnostic avoids BF16 emulation on this Zen3 host.
    net.initial_state=lambda b: tuple(v.view(1,1,512).expand(b,64,512) for v in (net.h_init,net.l_init))
    net.embeddings=lambda x,t: (512**.5*net.input(patchify(x).float()),net.time(t.reshape(-1,1).float())[:,None,:])
noise=torch.randn(2,3,32,32,generator=torch.Generator().manual_seed(5678)).to(device)
results=[]
@torch.no_grad()
def trajectory(label,checkpoint,steps,bypass=False):
    weights=dict(checkpoint['ema'])
    for name in ['refine.weight','refine.bias']:
        if name not in weights: weights[name]=torch.zeros_like(net.state_dict()[name])
    net.load_state_dict(weights)
    if bypass:
        net.refine.weight.zero_();net.refine.bias.zero_()
    y=noise.clone();state=net.initial_state(len(y));history=[]
    for i in range(steps):
        t=torch.full((len(y),),i/steps,device=device)
        captures=[]
        handle=net.refine.register_forward_hook(lambda m,args,out: captures.append((args[0].float().square().mean().sqrt().item(),out.float().square().mean().sqrt().item())))
        prediction,state=net(y,t,state)
        handle.remove()
        y=prediction if i==steps-1 else y+(prediction-y)/(steps-i)
        if i in [0,steps//2,steps-1]:
            history.append(dict(t=i/steps,y_std=y.std().item(),prediction_std=prediction.std().item(),
                                prediction_oob=(prediction.abs()>1).float().mean().item(),
                                image_rms=captures[0][0],residual_rms=captures[0][1]))
    r=dict(label=label,N=steps,mean=y.mean().item(),std=y.std().item(),oob_fraction=(y.abs()>1).float().mean().item(),
           min=y.min().item(),max=y.max().item(),history=history)
    results.append(r);print(json.dumps(r),flush=True)
trajectory('old_29000',old,32)
trajectory('refine_45000',new,32)
trajectory('refine_45000_bypass_head',new,32,True)

weights=new['ema']['refine.weight'];center=weights[:,:,1,1]
# Effective identity + residual convolution gain for constant RGB inputs.
dc=torch.eye(3)+weights.sum((2,3))
report=dict(samples=2,precision='FP32 CPU proxy' if device=='cpu' else 'original BF16',device=device,noise_seed=5678,head_weight_min=weights.min().item(),head_weight_max=weights.max().item(),
            head_center=center.tolist(),head_bias=new['ema']['refine.bias'].tolist(),effective_dc_gain=dc.tolist(),trajectories=results)
(base/'refine_diagnosis.json').write_text(json.dumps(report,indent=2))
print('head',json.dumps({k:v for k,v in report.items() if k!='trajectories'}),flush=True)
