"""Verify v4's carried state, shared noise, local gradients and stateful Euler."""
import copy
import json
import sys
from pathlib import Path
import torch
from adam_atan2 import AdamATan2
from model import StatefulTRMJiT, patchify, unpatchify, sample
from objective import sample_times, interpolate, weighted_mse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cifar10_looped_flows_v3_jit.model import TRMJiT


def main():
    torch.set_num_threads(3)
    torch.manual_seed(0); model=StatefulTRMJiT().cuda()
    torch.manual_seed(0); baseline=TRMJiT().cuda()  # Compare one cycle; compatible with the published v3 API.
    assert model.state_dict().keys()==baseline.state_dict().keys()
    assert all(torch.equal(model.state_dict()[k],baseline.state_dict()[k]) for k in model.state_dict())
    y=torch.randn(2,3,32,32,device='cuda'); noise=torch.randn_like(y)
    assert torch.equal(y,unpatchify(patchify(y)))
    generator=torch.Generator(device='cuda').manual_seed(12)
    times=sample_times(2,generator=generator)
    generator.manual_seed(12)
    draws=(torch.randn(2,4,device='cuda',generator=generator)*.8-.8).sigmoid()
    assert torch.equal(times,draws.sort(dim=1).values)
    assert torch.all(times[:,1:]>=times[:,:-1])
    with torch.no_grad():
        x=interpolate(y,noise,times[:,0])
        a,sa=model(x,times[:,0],model.initial_state(2))
        b,sb=baseline.forward_cycle(x,times[:,0],baseline.initial_state(2))
        assert torch.equal(a,b) and all(torch.equal(u,v) for u,v in zip(sa,sb))
    del baseline
    opt=AdamATan2(model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=1.)
    counts={False:0,True:0}
    def count(*_): counts[torch.is_grad_enabled()]+=1
    handle=model.core.register_forward_pre_hook(count)
    initial=tuple(s.clone().requires_grad_(True) for s in model.initial_state(2))
    state=initial
    rng_before=torch.cuda.get_rng_state().clone()
    for i in range(4):
        x=interpolate(y,noise,times[:,i])
        t=times[:,i,None,None,None]
        assert torch.allclose((x-t*y)/(1-t),noise,atol=3e-6,rtol=3e-6)
        opt.zero_grad(set_to_none=True)
        pred,state=model(x,times[:,i],state)
        assert all(not s.requires_grad for s in state)
        loss=weighted_mse(pred,y,times[:,i]); loss.backward()
        assert all(s.grad is None for s in initial)
        grad=sum(p.grad.abs().sum().item() for p in model.core.parameters() if p.grad is not None)
        assert grad==0 if i==0 else grad>0
        opt.step()
    handle.remove()
    assert counts=={False:56,True:28}
    assert torch.equal(rng_before,torch.cuda.get_rng_state())
    # Loaded state and optimizer must reproduce the next local update exactly.
    clone=copy.deepcopy(model)
    other=AdamATan2(clone.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=1.)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    for net,optimizer in ((model,opt),(clone,other)):
        optimizer.zero_grad(set_to_none=True)
        prediction,_=net(x,times[:,-1],state)
        weighted_mse(prediction,y,times[:,-1]).backward(); optimizer.step()
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),clone.parameters()))
    # Nonzero trained head must respond to a different incoming state.
    with torch.no_grad():
        carried,_=model(x,times[:,-1],state)
        reset,_=model(x,times[:,-1],model.initial_state(2))
        assert not torch.equal(carried,reset)
    class StatefulCounter:
        def __init__(self): self.inputs=[]; self.resets=0
        def initial_state(self,b): self.resets+=1; return 0
        def __call__(self,x,t,state):
            self.inputs.append(state)
            return torch.full_like(x,.1*(state+1)),state+1
    counter=StatefulCounter(); actual=sample(counter,noise,32)
    expected=noise.clone()
    for i in range(32): expected+=(.1*(i+1)-expected)/(32*max(1-i/32,.05))
    assert torch.equal(actual,expected)
    assert counter.inputs==list(range(32)) and counter.resets==1
    counter=StatefulCounter(); sample(counter,noise,8,reset_state=True)
    assert counter.inputs==[0]*8
    report=dict(initialization_and_single_cycle_equal_v3=True,parameters=sum(p.numel() for p in model.parameters()),
        K=4,Nsup_per_time=1,ordered_logitnormal_all_draws=True,shared_noise_reconstruction=True,
        no_noise_redraw_between_times=True,core_calls=counts,detached_temporal_state=True,
        zero_head_then_nonzero_core_gradient=True,optimizer_replay_exact=True,
        trained_state_influences_prediction=True,stateful_euler_all_steps=True,reset_state_ablation=True)
    out=Path('runs/cifar10_looped_flows_v4_stateful_jit');out.mkdir(parents=True,exist_ok=True)
    (out/'verification.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
