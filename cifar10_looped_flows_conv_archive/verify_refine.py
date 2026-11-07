"""Check real-checkpoint migration, exact zero-head identity, gradients and optimizer state."""
import copy
import json
from pathlib import Path
import tempfile
import torch
from torch import nn
from adam_atan2 import AdamATan2
from model import LoopedFlow
from migrate_refine import migrate
from data import ROOT


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_tree_equal(x, y)
    else:
        assert a == b


def main():
    torch.set_num_threads(3)
    torch.manual_seed(7)
    source = ROOT/'runs/cifar10_looped_flows/seed0/latest.pt'
    old = torch.load(source, map_location='cpu', weights_only=False)
    migrated = migrate(old, source)
    net = LoopedFlow().cuda()
    net.load_state_dict(migrated['model'])
    assert sum(p.numel() for p in net.refine.parameters()) == 84
    assert sum(p.numel() for p in net.parameters()) == 5129812
    assert_tree_equal(old['optimizer']['state'], migrated['optimizer']['state'])
    assert_tree_equal(old['rng'], migrated['rng'])
    assert_tree_equal(old['cuda_rng'], migrated['cuda_rng'])
    for key in ('model', 'ema'):
        for name in old[key]: assert_tree_equal(old[key][name], migrated[key][name])
        assert migrated[key]['refine.weight'].count_nonzero() == 0
        assert migrated[key]['refine.bias'].count_nonzero() == 0
    opt = AdamATan2(net.parameters(), lr=1e-4, betas=(.9,.95), weight_decay=1)
    opt.load_state_dict(migrated['optimizer'])
    for state in opt.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor): state[key] = value.cuda()
    assert len(opt.state) == len(list(net.parameters())) - 2
    baseline = LoopedFlow().cuda()
    baseline.load_state_dict(migrated['model'])
    class Zero(nn.Module):
        def forward(self, image): return torch.zeros_like(image)
    baseline.refine = Zero()
    baseline_opt = AdamATan2(baseline.parameters(), lr=1e-4, betas=(.9,.95), weight_decay=1)
    baseline_opt.load_state_dict(old['optimizer'])
    for state in baseline_opt.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor): state[key] = value.cuda()
    x = torch.randn(2,3,32,32,device='cuda')
    target = torch.randn_like(x)
    t = torch.tensor([.2,.8],device='cuda')
    prediction,_ = net(x,t,net.initial_state(2))
    expected,_ = baseline(x,t,baseline.initial_state(2))
    torch.testing.assert_close(prediction,expected,rtol=0,atol=0)
    (prediction-target).square().mean().backward()
    (expected-target).square().mean().backward()
    assert net.refine.weight.grad.abs().sum() > 0
    assert net.refine.bias.grad.abs().sum() > 0
    assert net.input.weight.grad.abs().sum() > 0
    for p in net.parameters(): assert torch.isfinite(p.grad).all()
    # At zero initialization, the first backbone update must match the old
    # optimizer exactly; moments/steps must not have been reset or remapped.
    opt.step(); baseline_opt.step()
    for name, p in baseline.named_parameters():
        torch.testing.assert_close(p,dict(net.named_parameters())[name],rtol=0,atol=0)
    assert net.refine.weight.count_nonzero() > 0
    assert opt.state[net.refine.weight]['step'].item() == 1
    report = dict(parent_step=old['step'], parameters=5129812, added_parameters=84,
                  initial_output_bitwise_equal=True, model_and_ema_preserved=True,
                  rng_preserved=True, old_optimizer_state_preserved=True,
                  first_backbone_update_bitwise_equal=True, refine_receives_gradient=True,
                  refine_first_optimizer_step=1, training_started=False)
    output=ROOT/'runs/cifar10_looped_flows/verification_refine.json'
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
