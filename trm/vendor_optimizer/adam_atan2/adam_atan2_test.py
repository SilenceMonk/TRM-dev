from typing import Tuple
import torch

import math
import pytest

from .adam_atan2 import _adam_atan2


def _adam_atan2_reference_impl(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    # Constant
    step_size: float,
    wd_step_size: float,
    bias_correction2_sqrt: float,
    beta1: float,
    beta2: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_dtype = param.dtype

    # Cast to math type, fp32.
    if orig_dtype != torch.float64:
        param = param.to(torch.float32)
        grad = grad.to(torch.float32)
        exp_avg = exp_avg.to(torch.float32)
        exp_avg_sq = exp_avg_sq.to(torch.float32)

    # Math
    # Reference implementation (PyTorch):
    # https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
    param.mul_(1 - wd_step_size)

    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    denom = exp_avg_sq.sqrt() / bias_correction2_sqrt
    param.add_(torch.atan2(exp_avg, denom), alpha=-step_size)

    return param.to(orig_dtype), exp_avg.to(orig_dtype), exp_avg_sq.to(orig_dtype)


@pytest.mark.parametrize("params_shape", [(1, ), (4096, ), (4096, 14336)])
@pytest.mark.parametrize("lr", [1e-3, 1e-4, 5e-4])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_adam_atan2_backend(
    params_shape,
    lr,
    dtype,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    init_std=0.02,
    grad_std=0.001,
    steps=100,
    atol={torch.float64: 1e-15, torch.float32: 1e-6, torch.float16: 0.002, torch.bfloat16: 0.005},
):
    torch.random.manual_seed(0)

    # Reference
    ref_param = torch.empty(params_shape, dtype=dtype, device="cuda").normal_(std=init_std)
    ref_exp_avg = torch.zeros_like(ref_param, dtype=dtype)
    ref_exp_avg_sq = torch.zeros_like(ref_param, dtype=dtype)
    ref_steps = 0

    # Test
    test_param = ref_param.clone()
    test_exp_avg = ref_exp_avg.clone()
    test_exp_avg_sq = ref_exp_avg_sq.clone()
    test_steps = torch.zeros((), dtype=torch.float32, device="cuda")

    for _ in range(steps):
        grad = torch.empty(params_shape, dtype=dtype, device="cuda").normal_(std=grad_std)

        # Reference
        ref_steps += 1
        ref_param, ref_exp_avg, ref_exp_avg_sq = _adam_atan2_reference_impl(
            ref_param,
            grad,
            ref_exp_avg,
            ref_exp_avg_sq,
            step_size=lr / (1 - beta1 ** ref_steps),
            wd_step_size=lr * weight_decay,
            bias_correction2_sqrt=math.sqrt(1 - beta2 ** ref_steps),
            beta1=beta1,
            beta2=beta2
        )

        # Test
        _adam_atan2(
            [test_param],
            [grad],
            [test_exp_avg],
            [test_exp_avg_sq],
            [test_steps],
            beta1,
            beta2,
            lr,
            weight_decay
        )

    # Check
    assert torch.allclose(test_param, ref_param, rtol=0, atol=atol[dtype])
    assert torch.allclose(test_exp_avg, ref_exp_avg, rtol=0, atol=atol[dtype])
    assert torch.allclose(test_exp_avg_sq, ref_exp_avg_sq, rtol=0, atol=atol[dtype])
    assert test_steps.item() == ref_steps
