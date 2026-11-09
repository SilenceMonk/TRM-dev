"""Check TRM parity, truncated gradients, JiT loss, stateless inference and Euler."""
import copy
import json
import sys
from pathlib import Path
import torch
from model import TRMJiT, AttentionBlock, patchify, unpatchify, sample
from objective import sample_times, weighted_mse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trm.model import TinyRecursiveReasoningModel_ACTV1Block, TinyRecursiveReasoningModel_ACTV1Config
from adam_atan2 import AdamATan2


def main():
    torch.set_num_threads(3)
    torch.manual_seed(17)
    device = 'cuda'
    model = TRMJiT().to(device)
    cfg = TinyRecursiveReasoningModel_ACTV1Config(batch_size=2, seq_len=64,
        num_puzzle_identifiers=1, vocab_size=1, H_cycles=3, L_cycles=6, H_layers=0,
        L_layers=2, hidden_size=512, expansion=4, num_heads=8, pos_encodings='rope',
        halt_max_steps=4, halt_exploration_prob=0, mlp_t=False, puzzle_emb_len=0)
    reference = TinyRecursiveReasoningModel_ACTV1Block(cfg).cuda()
    reference.load_state_dict(model.core.layers[0].state_dict())
    h = torch.randn(2,64,512, device=device, dtype=torch.bfloat16)
    assert torch.equal(model.core.layers[0](h, model.rotary_emb()), reference(model.rotary_emb(), h))
    x = torch.randn(2,3,32,32,device=device)
    assert torch.equal(x, unpatchify(patchify(x)))
    t = torch.tensor([.2,.99],device=device)
    target = torch.randn_like(x)
    z = t[:,None,None,None]*target + (1-t[:,None,None,None])*x
    prediction = torch.randn_like(x)
    d = (1-t).clamp_min(.05)[:,None,None,None]
    assert torch.allclose(weighted_mse(prediction,target,t), (((prediction-z)/d-(target-z)/d)**2).mean())
    gen = torch.Generator(device=device).manual_seed(12)
    times = sample_times(2, generator=gen)
    gen.manual_seed(12)
    assert torch.equal(times, (torch.randn(2,device=device,generator=gen)*.8-.8).sigmoid())
    counts = {False:0, True:0}
    def count(*_): counts[torch.is_grad_enabled()] += 1
    handle = model.core.register_forward_pre_hook(count)
    optimizer = AdamATan2(model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=1.)
    state = model.initial_state(2)
    for cycle in range(4):
        optimizer.zero_grad(set_to_none=True)
        pred,state = model.forward_cycle(z,t,state)
        assert all(not v.requires_grad for v in state)
        loss = weighted_mse(pred,target,t)
        loss.backward()
        core_grad = sum(p.grad.abs().sum().item() for p in model.core.parameters() if p.grad is not None)
        assert core_grad == 0 if cycle == 0 else core_grad > 0
        optimizer.step()
    handle.remove()
    assert counts == {False:56, True:28}, counts
    with torch.no_grad():
        state = model.initial_state(2)
        for _ in range(4): pred,state = model.forward_cycle(z,t,state)
        assert torch.equal(pred, model(z,t))
        model(x,t)
        assert torch.equal(pred, model(z,t))
    # Optimizer state roundtrip must reproduce the next supervised update.
    clone = copy.deepcopy(model)
    other = AdamATan2(clone.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=1.)
    other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for net,opt in ((model,optimizer),(clone,other)):
        opt.zero_grad(set_to_none=True)
        weighted_mse(net(z,t),target,t).backward()
        opt.step()
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),clone.parameters()))
    class Constant(torch.nn.Module):
        def forward(self,x,t): return torch.ones_like(x)*.7
    expected = x.float().clone()
    for i in range(32): expected += (.7-expected)/(32*max(1-i/32,.05))
    assert torch.allclose(sample(Constant(),x,32),expected)
    report = dict(attention_matches_original=True, patch_roundtrip=True, jit_loss_equivalence=True,
        single_time_sampling=True, Nsup=4, core_calls=counts, detached_state=True,
        zero_head_then_nonzero_core_gradient=True, stateless_four_cycle_prediction=True,
        optimizer_replay_exact=True, clamped_euler_all_steps=True,
        parameters=sum(p.numel() for p in model.parameters()), torch=torch.__version__)
    out = Path('runs/cifar10_looped_flows_v3_jit'); out.mkdir(parents=True,exist_ok=True)
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
