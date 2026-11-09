"""Direct JiT: one independent logit-normal time per image, no ordering."""
import torch


def sample_times(batch_size, device='cuda', mean=-0.8, std=0.8, generator=None):
    return torch.sigmoid(torch.randn(batch_size, device=device, generator=generator)*std + mean)


def weighted_mse(prediction, target, t, t_eps=0.05):
    if t_eps <= 0:
        raise ValueError('t_eps must be positive')
    denominator = (1-t.float()).clamp_min(t_eps).view(-1, 1, 1, 1)
    return ((prediction.float()-target.float())/denominator).square().mean()
