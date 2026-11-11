"""Causality, shifted labels, gradients, and optimizer continuation checks."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import numpy as np
import torch
try:
    from .runtime import ROOT, ModelConfig, atomic_json, setup
    from .model import TRMLanguageModel
    from .train import build, update
    from .data import TokenDataset
except ImportError:
    from runtime import ROOT, ModelConfig, atomic_json, setup
    from model import TRMLanguageModel
    from train import build, update
    from data import TokenDataset


def main():
    setup()
    torch.manual_seed(123)
    cfg = ModelConfig(seq_len=32, vocab_size=512)
    model, opt = build(cfg, compile_mode=False)
    x = torch.randint(512, (2, 32), device='cuda')
    y = torch.randint(512, (2, 32), device='cuda')
    changed = x.clone()
    changed[:, 17:] = (changed[:, 17:] + 37) % cfg.vocab_size
    with torch.no_grad():
        original = model(x)
        perturbed = model(changed)
    torch.testing.assert_close(original[:, :17], perturbed[:, :17], atol=0, rtol=0)
    assert (original[:, 17:] - perturbed[:, 17:]).abs().max() > .01
    loss, norm = update(model, opt, [(x, y)])
    assert torch.isfinite(loss) and torch.isfinite(norm)
    assert model.embedding.embedding_weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.core.parameters())
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        np.arange(100, dtype='<u2').tofile(temp / 'train.bin')
        dataset = TokenDataset(temp, 'train', 8)
        inputs, labels = dataset.batch([0, 3])
        assert inputs.tolist() == [list(range(8)), list(range(24, 32))]
        assert labels.tolist() == [list(range(1, 9)), list(range(25, 33))]
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), temp / 'resume.pt')
        expected_loss, _ = update(model, opt, [(x, y)])
        saved = torch.load(temp / 'resume.pt', map_location='cpu', weights_only=False)
        resumed, resumed_opt = build(cfg, compile_mode=False)
        resumed.load_state_dict(saved['model'])
        resumed_opt.load_state_dict(saved['optimizer'])
        for state in resumed_opt.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.cuda()
        actual_loss, _ = update(resumed, resumed_opt, [(x, y)])
        torch.testing.assert_close(expected_loss, actual_loss, atol=0, rtol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
    report = dict(causal_prefix_bitwise_exact=True, next_token_shift_correct=True,
        finite_forward_backward=True, shared_core_gradients=True,
        optimizer_resume_bitwise_exact=True, model_config=asdict(cfg))
    atomic_json(ROOT / 'runs/trm_tinystories/verification.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
