import math

import torch
from torch import nn


class BoundedLogVariance(nn.Module):
    """Predict log variance inside fixed smooth bounds."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        log_var_min: float = -10.0,
        log_var_max: float = 5.0,
    ) -> None:
        super().__init__()
        if log_var_min >= log_var_max:
            raise ValueError('log_var_min must be smaller than log_var_max')

        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)
        self.proj = nn.Linear(input_dim, output_dim)

        # sigmoid(raw)=(-log_var_min)/(log_var_max-log_var_min) gives log_var=0
        initial_probability = -self.log_var_min / (
            self.log_var_max - self.log_var_min
        )
        initial_bias = math.log(
            initial_probability / (1.0 - initial_probability)
        )
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, initial_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_log_var = self.proj(x)
        scale = self.log_var_max - self.log_var_min
        return self.log_var_min + scale * torch.sigmoid(raw_log_var)


def reparameterize(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    epsilon: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw a differentiable sample from a diagonal Gaussian."""
    if epsilon is None:
        epsilon = torch.randn_like(mean)
    if epsilon.shape != mean.shape:
        raise ValueError(
            f'epsilon shape {epsilon.shape} does not match mean {mean.shape}'
        )
    return mean + torch.exp(0.5 * log_var) * epsilon


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    """Elementwise negative log likelihood for a diagonal Gaussian."""
    target, mean, log_var = target.float(), mean.float(), log_var.float()
    return 0.5 * (
        math.log(2.0 * math.pi)
        + log_var
        + (target - mean).square() * torch.exp(-log_var)
    )


def unit_gaussian_kl(
    mean: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """Elementwise KL(N(mean, var) || N(0, I))."""
    mean, log_var = mean.float(), log_var.float()
    return 0.5 * (torch.exp(log_var) + mean.square() - 1.0 - log_var)


__all__ = [
    'BoundedLogVariance',
    'gaussian_nll',
    'reparameterize',
    'unit_gaussian_kl',
]
