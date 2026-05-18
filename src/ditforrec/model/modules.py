from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return self.mlp(embedding)


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(sequence.dtype)
    summed = (sequence * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return summed / denom


class CondLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, timestep_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.condition = nn.Sequential(
            nn.Linear(hidden_dim + timestep_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, x: torch.Tensor, timestep_emb: torch.Tensor, history_context: torch.Tensor) -> torch.Tensor:
        conditioned = self.condition(torch.cat([timestep_emb, history_context], dim=-1))
        scale, shift = conditioned.chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CrossAttentionAdapter(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = cond_mask == 0
        normalized = self.norm(x)
        attended, _ = self.attn(normalized, cond, cond, key_padding_mask=key_padding_mask, need_weights=False)
        return x + attended


class HistoryTimeCorrection(nn.Module):
    def __init__(self, hidden_dim: int, timestep_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + timestep_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        clean_history: torch.Tensor,
        history_mask: torch.Tensor,
        timestep_emb: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm(query_tokens)
        key_padding_mask = history_mask == 0
        attended, _ = self.attn(normalized, clean_history, clean_history, key_padding_mask=key_padding_mask, need_weights=False)
        history_context = masked_mean(clean_history, history_mask)
        gate = self.gate(torch.cat([history_context, timestep_emb], dim=-1)).unsqueeze(1)
        return query_tokens + gate * attended


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
