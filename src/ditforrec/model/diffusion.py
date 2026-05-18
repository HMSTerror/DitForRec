from __future__ import annotations

import math

import torch
from torch import nn


class GaussianDiffusionScheduler(nn.Module):
    def __init__(self, num_steps: int = 50, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alpha_bars", torch.sqrt(1.0 / alpha_bars))
        self.register_buffer("sqrt_recipm1_alpha_bars", torch.sqrt(torch.clamp(1.0 / alpha_bars - 1.0, min=0.0)))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars))
        self.register_buffer("posterior_mean_coef2", (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars))

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
        return values[timesteps].view(shape)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, timesteps, x0)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, x0)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def predict_noise_from_start(self, xt: torch.Tensor, timesteps: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, timesteps, xt)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, xt)
        return (xt - sqrt_alpha_bar * x0) / sqrt_one_minus.clamp(min=1e-8)

    def q_posterior(self, x_start: torch.Tensor, xt: torch.Tensor, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = self._extract(self.posterior_mean_coef1, timesteps, xt) * x_start
        posterior_mean = posterior_mean + self._extract(self.posterior_mean_coef2, timesteps, xt) * xt
        posterior_variance = self._extract(self.posterior_variance, timesteps, xt)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, timesteps, xt)
        return posterior_mean, posterior_variance, posterior_log_variance

    def posterior_sample(self, xt: torch.Tensor, x_start: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        posterior_mean, _, posterior_log_variance = self.q_posterior(x_start, xt, timesteps)
        noise = torch.randn_like(xt)
        nonzero_mask = (timesteps > 0).float().view((xt.shape[0],) + (1,) * (xt.ndim - 1))
        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

    def ddim_step(
        self,
        xt: torch.Tensor,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        prev_timesteps: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        eps = self.predict_noise_from_start(xt, timesteps, x_start)
        alpha_bar_t = self._extract(self.alpha_bars, timesteps, xt)

        raw_prev_timesteps = prev_timesteps
        prev_timesteps = prev_timesteps.clamp(min=0)
        alpha_bar_prev = self._extract(self.alpha_bars, prev_timesteps, xt)
        sigma = 0.0
        if eta > 0.0:
            sigma = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp(min=1e-8))
            sigma = sigma * torch.sqrt((1.0 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8)).clamp(min=0.0))

        noise = torch.randn_like(xt) if eta > 0.0 else torch.zeros_like(xt)
        direction = torch.sqrt((1.0 - alpha_bar_prev - sigma**2).clamp(min=0.0)) * eps
        sample = torch.sqrt(alpha_bar_prev) * x_start + direction + sigma * noise

        final_mask = (raw_prev_timesteps < 0).float().view((xt.shape[0],) + (1,) * (xt.ndim - 1))
        return final_mask * x_start + (1.0 - final_mask) * sample

    def prior_matching_loss(self, x0: torch.Tensor) -> torch.Tensor:
        alpha_bar_final = float(self.alpha_bars[-1].item())
        sigma_sq = max(1.0 - alpha_bar_final, 1e-8)
        mu_sq = alpha_bar_final * (x0**2)
        kl = 0.5 * (mu_sq + sigma_sq - 1.0 - math.log(sigma_sq))
        return kl.mean()
