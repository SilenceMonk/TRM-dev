"""Per-token stopping, frozen states, causal recurrence, local supervision and EMA resume."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import torch
try:
    from .runtime import ROOT, atomic_json, setup
    from .model_act import ACTConfig, Carry, TokenACT
    from .train_act import build, deep_update
except ImportError:
    from runtime import ROOT, atomic_json, setup
    from model_act import ACTConfig, Carry, TokenACT
    from train_act import build, deep_update


def main():
    setup()
    torch.manual_seed(7)
    cfg = ACTConfig(seq_len=12, vocab_size=128)
    model = TokenACT(cfg).cuda().eval()
    x = torch.randint(cfg.vocab_size, (2, 12), device='cuda')
    with torch.no_grad():
        model.q_head.bias.fill_(100)
        _, steps = model.forward_with_steps(x)
        assert (steps == 1).all()
        model.q_head.bias.fill_(-100)
        _, steps = model.forward_with_steps(x)
        assert (steps == 16).all()
        model.q_head.weight.normal_(0, .3)
        model.q_head.bias.zero_()
        changed = x.clone()
        changed[:, 7:] = (changed[:, 7:] + 13) % cfg.vocab_size
        pred, steps = model.forward_with_steps(x)
        other, other_steps = model.forward_with_steps(changed)
        torch.testing.assert_close(pred[:, :7], other[:, :7], atol=0, rtol=0)
        torch.testing.assert_close(steps[:, :7], other_steps[:, :7], atol=0, rtol=0)
        assert steps.unique().numel() > 1, 'Test must exercise different token depths'
    carry = model.initial_carry(x)
    mask = torch.arange(x.shape[1], device='cuda')[None, :].expand_as(x) % 2 == 0
    carry = Carry(carry.h.clone(), carry.l.clone(), mask.int(), mask)
    proposed, _, _ = model.forward_cycle(x, carry)
    torch.testing.assert_close(proposed.h[mask], carry.h[mask], atol=0, rtol=0)
    torch.testing.assert_close(proposed.l[mask], carry.l[mask], atol=0, rtol=0)
    assert not proposed.h.requires_grad and not proposed.l.requires_grad
    q = torch.full(x.shape, 100., device='cuda')
    for _ in range(4):
        proposed, _, _ = model.forward_cycle(x, carry)
        carry = model.finish_cycle(proposed, q)
    assert (carry.steps[mask] == 1).all()
    # Small horizon for an exact optimizer/EMA/RNG continuation test, original H/L retained.
    cfg = ACTConfig(seq_len=12, vocab_size=128, halt_max_steps=3)
    torch.manual_seed(31)
    raw, optimizer = build(cfg, 1e-4)
    ema = copy.deepcopy(raw).eval().requires_grad_(False)
    y = torch.randint(128, x.shape, device='cuda')
    batches = [(x, y)]
    result = deep_update(raw, optimizer, ema, batches, 0, 1e-4, 1, .999)
    assert result[2] == 3 and all(s['step'].item() == 3 for s in optimizer.state.values())
    assert raw.q_head.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in raw.core.parameters())
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'resume.pt'
        torch.save(dict(model=raw.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                        rng=torch.cuda.get_rng_state_all()), path)
        expected = deep_update(raw, optimizer, ema, batches, 3, 1e-4, 1, .999)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        replay, opt = build(cfg, 1e-4)
        replay_ema = copy.deepcopy(replay).eval().requires_grad_(False)
        replay.load_state_dict(saved['model'])
        replay_ema.load_state_dict(saved['ema'])
        opt.load_state_dict(saved['optimizer'])
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        torch.cuda.set_rng_state_all(saved['rng'])
        actual = deep_update(replay, opt, replay_ema, batches, 3, 1e-4, 1, .999)
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        for left, right in ((raw, replay), (ema, replay_ema)):
            for k, v in left.state_dict().items():
                torch.testing.assert_close(v, right.state_dict()[k], atol=0, rtol=0)
    report = dict(token_depth_varies=True, maximum_cycles_16=True, immediate_halt=True,
        causal_prefix_logits_and_halts_bitwise_exact=True, halted_states_frozen=True,
        carry_detached=True, halt_head_gradients=True, core_gradients_finite=True,
        optimizer_steps_per_supervision_cycle=True, raw_ema_rng_resume_bitwise_exact=True)
    atomic_json(ROOT / 'runs/trm_tinystories/verification_act.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
