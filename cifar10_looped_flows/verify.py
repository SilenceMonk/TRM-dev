"""Executable invariants: patch order, gradient horizon, optimizer replay, Euler endpoint."""
import copy
import argparse
import json
from pathlib import Path
import tempfile
import torch
from adam_atan2 import AdamATan2
from model import LoopedFlow, patchify, unpatchify, sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/cifar10_looped_flows/verification.json')
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(0)
    x = torch.arange(2*3*32*32).reshape(2,3,32,32)
    assert torch.equal(unpatchify(patchify(x)), x)
    assert torch.equal(patchify(x)[0,0], x[0,:,:4,:4].reshape(-1))
    model = LoopedFlow().cuda()
    x = torch.randn(2,3,32,32,device='cuda')
    t = torch.tensor([.2,.8],device='cuda')
    h, l = (v.clone().requires_grad_() for v in model.initial_state(2))
    calls = []
    handle = model.core.register_forward_hook(lambda *_: calls.append(torch.is_grad_enabled()))
    pred, state = model(x, t, (h,l))
    handle.remove()
    assert calls == [False]*14+[True]*7, calls
    assert not any(v.requires_grad for v in state)
    loss = pred.square().mean()
    loss.backward()
    assert h.grad is None and l.grad is None
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.input.weight.grad.abs().sum() > 0
    assert model.time[0].weight.grad.abs().sum() > 0
    optimizer = AdamATan2(model.parameters(), lr=1e-4, betas=(.9,.95),weight_decay=1.)
    optimizer.step()
    # Exact next-update replay through serialized model, optimizer, and RNG.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'checkpoint.pt'
        torch.save(dict(model=model.state_dict(), opt=optimizer.state_dict(),rng=torch.cuda.get_rng_state()),path)
        clone = LoopedFlow().cuda()
        other = AdamATan2(clone.parameters(), lr=1e-4, betas=(.9,.95),weight_decay=1.)
        saved = torch.load(path,weights_only=False)
        clone.load_state_dict(saved['model'])
        other.load_state_dict(saved['opt'])
        for state_value in other.state.values():
            for k,v in state_value.items():
                if isinstance(v, torch.Tensor): state_value[k]=v.cuda()
        def update(net,opt):
            opt.zero_grad(set_to_none=True)
            y = torch.randn_like(x)
            pred,_ = net(y,t,net.initial_state(2))
            pred.square().mean().backward()
            opt.step()
        torch.cuda.set_rng_state(saved['rng']); update(model,optimizer)
        torch.cuda.set_rng_state(saved['rng']); update(clone,other)
        for a,b in zip(model.parameters(),clone.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    class Constant:
        def initial_state(self,b): return None
        def __call__(self,y,t,state):
            assert (t<1).all()
            return torch.full_like(y,.25), state
    torch.testing.assert_close(sample(Constant(),x,8),torch.full_like(x,.25),rtol=0,atol=0)
    report=dict(parameters=sum(p.numel() for p in model.parameters()), patch_roundtrip=True,
                gradient_calls=calls, finite_all_gradients=True, detached_boundaries=True,
                adam_atan2_exact_resume=True, euler_endpoint=True)
    print(json.dumps(report,indent=2))
    out = Path(__file__).resolve().parents[1]/args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))

if __name__ == '__main__': main()
