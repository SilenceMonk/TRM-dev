"""JiT objective with ordered logit-normal times and a shared noise-target pair."""
import torch


def sample_times(batch_size, K=4, device='cuda', mean=-0.8, std=0.8, generator=None):
    if K < 1 or std <= 0:
        raise ValueError('K and std must be positive')
    draws = torch.sigmoid(torch.randn(batch_size, K, device=device, generator=generator)*std + mean)
    return draws.sort(dim=1).values


def interpolate(target, noise, t):
    t = t.float().view(-1,1,1,1)
    return t*target + (1-t)*noise


def weighted_mse(prediction, target, t, t_eps=0.05):
    if t_eps <= 0:
        raise ValueError('t_eps must be positive')
    denominator = (1-t.float()).clamp_min(t_eps).view(-1, 1, 1, 1)
    return ((prediction.float()-target.float())/denominator).square().mean()
