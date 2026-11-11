"""EMA equation, exact raw/EMA optimizer continuation, and epoch plateau decisions."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import torch
try:
    from .runtime import ROOT, ModelConfig, atomic_json, setup
    from .train import build, update
    from .train_ema import Plateau, update_ema
except ImportError:
    from runtime import ROOT, ModelConfig, atomic_json, setup
    from train import build, update
    from train_ema import Plateau, update_ema


def main():
    setup()
    torch.manual_seed(17)
    cfg = ModelConfig(seq_len=32, vocab_size=512)
    model, optimizer = build(cfg, compile_mode=False)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    x = torch.randint(512, (2, 32), device='cuda')
    y = torch.randint(512, (2, 32), device='cuda')
    initial = {k: v.clone() for k, v in model.named_parameters()}
    update(model, optimizer, [(x, y)])
    update_ema(model, ema, .999)
    for k, v in ema.named_parameters():
        expected = initial[k] * .999 + dict(model.named_parameters())[k] * .001
        torch.testing.assert_close(v, expected, atol=2e-8, rtol=1e-6)
        assert v.grad is None and not v.requires_grad
    for k, v in ema.named_buffers():
        torch.testing.assert_close(v, dict(model.named_buffers())[k], atol=0, rtol=0)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'resume.pt'
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict()), path)
        expected_loss, _ = update(model, optimizer, [(x, y)])
        update_ema(model, ema, .999)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        replay, replay_opt = build(cfg, compile_mode=False)
        replay_ema = copy.deepcopy(replay).eval().requires_grad_(False)
        replay.load_state_dict(saved['model'])
        replay_ema.load_state_dict(saved['ema'])
        replay_opt.load_state_dict(saved['optimizer'])
        for state in replay_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        actual_loss, _ = update(replay, replay_opt, [(x, y)])
        update_ema(replay, replay_ema, .999)
        torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
        for left, right in ((model, replay), (ema, replay_ema)):
            for k, v in left.state_dict().items():
                torch.testing.assert_close(v, right.state_dict()[k], atol=0, rtol=0)
    control = Plateau(lr=3e-4)
    assert control.observe(2., 1)
    assert not control.observe(1.9995, 2) and control.stale_epochs == 1
    control = Plateau(**asdict(control))
    assert not control.observe(2.01, 3) and not control.stopped
    assert not control.observe(2., 4) and control.stopped
    assert control.lr == 3e-4 / 8
    reset = Plateau(lr=3e-4)
    reset.observe(2., 1)
    reset.observe(2.1, 2)
    assert reset.observe(1.998, 3) and reset.stale_epochs == 0 and not reset.stopped
    report = dict(ema_equation=True, ema_has_no_grad=True, fixed_buffers_preserved=True,
        raw_ema_optimizer_resume_bitwise_exact=True, plateau_patience_and_lr=True,
        plateau_resume=True, significant_improvement_resets_patience=True)
    atomic_json(ROOT / 'runs/trm_tinystories/verification_ema.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
