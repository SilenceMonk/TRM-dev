"""Numerical invariants for JiT loss/head and stateful clamped Euler integration."""
import copy
import json
from pathlib import Path
import torch
from adam_atan2 import AdamATan2
from model import LoopedFlow, patchify, unpatchify, sample
from objective import weighted_mse, sample_times
from jit_head import FinalLayer
from data import ROOT

torch.set_num_threads(3)
torch.manual_seed(0)
x=torch.arange(2*3*32*32).reshape(2,3,32,32)
assert torch.equal(unpatchify(patchify(x)),x)
# Both sides of the official v loss use the same capped denominator.
t=torch.tensor([0.,.5,.99,1.],device='cuda')
y=torch.randn(4,3,32,32,device='cuda');noise=torch.randn_like(y);pred=torch.randn_like(y)
z=t[:,None,None,None]*y+(1-t[:,None,None,None])*noise
d=(1-t).clamp_min(.05)[:,None,None,None]
reference=((y-z)/d-(pred-z)/d).square().mean()
torch.testing.assert_close(weighted_mse(pred,y,t),reference)
assert torch.isfinite(weighted_mse(pred,y,t))
g1=torch.Generator().manual_seed(9);g2=torch.Generator().manual_seed(9)
expected=torch.sigmoid(torch.randn(1000,5,generator=g1)*.8-.8).sort(dim=1).values[:,:4]
torch.testing.assert_close(sample_times(1000,device='cpu',generator=g2),expected,rtol=0,atol=0)
assert (expected[:,1:]>=expected[:,:-1]).all()
# Independently execute the pinned upstream definitions; no optional timm/rope imports.
src=Path(__file__).parent/'reference'
model_src=(src/'model_jit.py').read_text();util_src=(src/'model_util.py').read_text()
namespace=dict(torch=torch,nn=torch.nn)
exec(model_src[model_src.index('def modulate('):model_src.index('\n\nclass BottleneckPatchEmbed')],namespace)
exec(util_src[util_src.index('class RMSNorm('):util_src.index('\n\ndef get_2d_sincos_pos_embed')],namespace)
exec(model_src[model_src.index('class FinalLayer('):model_src.index('\n\nclass JiTBlock')].replace('    @torch.compile\n',''),namespace)
head=FinalLayer(512,4,3).cuda();official=namespace['FinalLayer'](512,4,3).cuda()
official.load_state_dict(head.state_dict())
h=torch.randn(2,64,512,device='cuda');c=torch.randn(2,512,device='cuda')
torch.testing.assert_close(head(h,c),official(h,c),rtol=0,atol=0)
net=LoopedFlow().cuda();opt=AdamATan2(net.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=1)
x=torch.randn(2,3,32,32,device='cuda');target=torch.randn_like(x);time=torch.tensor([.2,.8],device='cuda')
calls=[]
handle=net.core.register_forward_hook(lambda *_:calls.append(torch.is_grad_enabled()))
pred,state=net(x,time,net.initial_state(2));handle.remove()
assert pred.count_nonzero()==0
assert calls==[False]*14+[True]*7
assert not any(v.requires_grad for v in state)
for i in range(2):
 opt.zero_grad(set_to_none=True)
 pred,_=net(x,time,net.initial_state(2))
 weighted_mse(pred,target,time).backward()
 for name,p in net.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all(),name
 if i==0:
  assert net.input.weight.grad.count_nonzero()==0
  assert net.output.linear.weight.grad.abs().sum()>0
 else:
  assert net.input.weight.grad.abs().sum()>0
  assert net.output.adaLN_modulation[-1].weight.grad.abs().sum()>0
 opt.step()
# AdamATan2 complete next-update replay.
clone=LoopedFlow().cuda();clone.load_state_dict(net.state_dict())
other=AdamATan2(clone.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=1)
other.load_state_dict(copy.deepcopy(opt.state_dict()))
for m,o in [(net,opt),(clone,other)]:
 o.zero_grad(set_to_none=True);p,_=m(x,time,m.initial_state(2));weighted_mse(p,target,time).backward();o.step()
for a,b in zip(net.parameters(),clone.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
class Constant:
 def initial_state(self,b):return None
 def __call__(self,y,t,state):
  assert (t<1).all();return torch.full_like(y,.25),state
for N in [8,32,64]:
 expected=x.clone()
 for i in range(N):expected=expected+(.25-expected)/(N*max(1-i/N,.05))
 actual=sample(Constant(),x,N,t_eps=.05)
 torch.testing.assert_close(actual,expected,rtol=0,atol=0)
 if N==64:assert not torch.equal(actual,torch.full_like(x,.25))
report=dict(parameters=sum(p.numel() for p in net.parameters()),weighted_loss_matches_official=True,
 logit_normal_ordered_times=True,official_output_head_bitwise_equal=True,initial_prediction_zero=True,
 first_step_zero_backbone_gradient_expected=True,second_step_backbone_gradient_nonzero=True,
 finite_all_gradients=True,gradient_calls=calls,optimizer_exact_replay=True,
 capped_euler_matches_reference=True,final_step_respects_floor=True)
path=ROOT/'runs/cifar10_looped_flows_v2_jit/verification.json';path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
