"""JiT x-prediction/v-loss and logit-normal times, adapted to ordered Looped Flows."""
import torch


def sample_times(batch_size, K=4, device='cuda', mean=-0.8, std=0.8, generator=None):
    draws = torch.sigmoid(torch.randn(batch_size, K+1, device=device, generator=generator)*std + mean)
    return draws.sort(dim=1).values[:, :K]


def weighted_mse(prediction, target, t, t_eps=0.05):
    if t_eps <= 0:
        raise ValueError('t_eps must be positive')
    denominator = (1-t.float()).clamp_min(t_eps).view(-1, 1, 1, 1)
    return ((prediction.float()-target.float())/denominator).square().mean()
