from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ditforrec.model.diffusion import GaussianDiffusionScheduler
from ditforrec.model.modules import (
    CondLayerNorm,
    CrossAttentionAdapter,
    FeedForward,
    HistoryTimeCorrection,
    SinusoidalTimeEmbedding,
    masked_mean,
)


@dataclass
class DitForRecOutput:
    loss: torch.Tensor
    denoise_loss: torch.Tensor
    target_recon_loss: torch.Tensor
    prior_loss: torch.Tensor
    ce_loss: torch.Tensor
    direct_ce_loss: torch.Tensor
    logits: torch.Tensor
    pred_target: torch.Tensor


class DCDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        timestep_dim: int,
        use_text_cross_attn: bool,
        use_image_cross_attn: bool,
        use_history_correction: bool,
    ) -> None:
        super().__init__()
        self.cond_ln = CondLayerNorm(hidden_dim, timestep_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.post_attn_norm = nn.LayerNorm(hidden_dim)
        self.text_cross = CrossAttentionAdapter(hidden_dim, num_heads, dropout) if use_text_cross_attn else None
        self.image_cross = CrossAttentionAdapter(hidden_dim, num_heads, dropout) if use_image_cross_attn else None
        self.explicit_correction = (
            HistoryTimeCorrection(hidden_dim, timestep_dim, num_heads, dropout) if use_history_correction else None
        )
        self.ffn = FeedForward(hidden_dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        timestep_emb: torch.Tensor,
        clean_history: torch.Tensor,
        history_mask: torch.Tensor,
        text_cond: Optional[torch.Tensor],
        image_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        history_context = masked_mean(clean_history, history_mask)
        conditioned = self.cond_ln(x, timestep_emb, history_context)
        self_attended, _ = self.self_attn(
            conditioned,
            conditioned,
            conditioned,
            key_padding_mask=token_mask == 0,
            need_weights=False,
        )
        x = x + self_attended
        x = self.post_attn_norm(x)

        if self.text_cross is not None:
            x = self.text_cross(x, text_cond, history_mask)
        if self.image_cross is not None:
            x = self.image_cross(x, image_cond, history_mask)

        if self.explicit_correction is not None:
            x = self.explicit_correction(x, clean_history, history_mask, timestep_emb)
        x = self.ffn(x)
        return x


class DitForRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_users: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_history: int = 50,
        num_diffusion_steps: int = 50,
        text_dim: int = 512,
        image_dim: int = 768,
        text_inject_layers: list[int] | None = None,
        image_inject_layers: list[int] | None = None,
        timestep_dim: int = 128,
        use_user_embeddings: bool = True,
        use_text_condition: bool = True,
        use_image_condition: bool = True,
        use_history_correction: bool = True,
        use_final_correction: bool = True,
        add_user_to_target: bool = True,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 2e-2,
        denoise_weight: float = 1.0,
        target_recon_weight: float = 0.5,
        prior_weight: float = 1e-4,
        ce_weight: float = 1.0,
        direct_ce_weight: float = 0.0,
        direct_score_weight: float = 0.0,
        label_smoothing: float = 0.0,
        logit_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        text_inject_layers = text_inject_layers or ([1] if use_text_condition else [])
        image_inject_layers = image_inject_layers or ([2] if use_image_condition else [])

        self.hidden_dim = hidden_dim
        self.max_history = max_history
        self.denoise_weight = denoise_weight
        self.target_recon_weight = target_recon_weight
        self.prior_weight = prior_weight
        self.ce_weight = ce_weight
        self.direct_ce_weight = direct_ce_weight
        self.direct_score_weight = min(max(direct_score_weight, 0.0), 1.0)
        self.label_smoothing = min(max(label_smoothing, 0.0), 1.0)
        self.logit_temperature = max(logit_temperature, 1e-6)
        self.use_text_condition = use_text_condition
        self.use_image_condition = use_image_condition
        self.use_final_correction = use_final_correction
        self.add_user_to_target = add_user_to_target
        self.item_embeddings = nn.Embedding(num_items, hidden_dim, padding_idx=0)
        self.user_embeddings = nn.Embedding(num_users, hidden_dim, padding_idx=0) if use_user_embeddings else None
        self.position_embeddings = nn.Parameter(torch.randn(1, max_history + 1, hidden_dim) * 0.02)
        self.direct_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        self.text_projector = nn.Linear(text_dim, hidden_dim) if use_text_condition else None
        self.image_projector = nn.Linear(image_dim, hidden_dim) if use_image_condition else None
        self.timestep_embedder = SinusoidalTimeEmbedding(timestep_dim)
        self.scheduler = GaussianDiffusionScheduler(
            num_steps=num_diffusion_steps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
        )

        self.blocks = nn.ModuleList(
            [
                DCDiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    timestep_dim=timestep_dim,
                    use_text_cross_attn=idx in text_inject_layers,
                    use_image_cross_attn=idx in image_inject_layers,
                    use_history_correction=use_history_correction,
                )
                for idx in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_correction = (
            HistoryTimeCorrection(hidden_dim, timestep_dim, num_heads, dropout) if use_final_correction else None
        )

    def build_history_tokens(self, user_ids: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        history_emb = self.item_embeddings(history)
        if self.user_embeddings is not None:
            non_padding = history.ne(0).unsqueeze(-1).to(history_emb.dtype)
            history_emb = (history_emb + self.user_embeddings(user_ids).unsqueeze(1)) * non_padding
        return history_emb

    def build_target_tokens(self, user_ids: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_emb = self.item_embeddings(target).unsqueeze(1)
        if self.user_embeddings is not None and self.add_user_to_target:
            target_emb = target_emb + self.user_embeddings(user_ids).unsqueeze(1)
        return target_emb

    def build_clean_tokens(self, user_ids: torch.Tensor, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        history_tokens = self.build_history_tokens(user_ids, history)
        target_tokens = self.build_target_tokens(user_ids, target)
        return torch.cat([history_tokens, target_tokens], dim=1)

    def denoise(
        self,
        noisy_tokens: torch.Tensor,
        history_mask: torch.Tensor,
        text_cond: Optional[torch.Tensor],
        image_cond: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        clean_history: torch.Tensor,
    ) -> torch.Tensor:
        x = noisy_tokens + self.position_embeddings[:, : noisy_tokens.shape[1], :]
        x = self.input_dropout(x)
        timestep_emb = self.timestep_embedder(timesteps)

        projected_text = self.text_projector(text_cond) if self.text_projector is not None else None
        projected_image = self.image_projector(image_cond) if self.image_projector is not None else None
        token_mask = torch.cat([history_mask, torch.ones_like(history_mask[:, :1])], dim=1)

        for block in self.blocks:
            x = block(x, token_mask, timestep_emb, clean_history, history_mask, projected_text, projected_image)

        x = self.final_norm(x)
        if self.final_correction is not None:
            corrected_target = self.final_correction(x[:, -1:, :], clean_history, history_mask, timestep_emb)
            x = torch.cat([x[:, :-1, :], corrected_target], dim=1)
        return x

    def compute_logits(self, target_repr: torch.Tensor) -> torch.Tensor:
        normalized_target = F.normalize(target_repr, dim=-1)
        normalized_items = F.normalize(self.item_embeddings.weight, dim=-1)
        return (normalized_target @ normalized_items.transpose(0, 1)) / self.logit_temperature

    def direct_score_logits(
        self,
        user_id: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        text_cond: torch.Tensor,
        image_cond: torch.Tensor,
    ) -> torch.Tensor:
        clean_history = self.build_history_tokens(user_id, history)
        query = self.direct_query.expand(history.shape[0], -1, -1)
        if self.user_embeddings is not None:
            query = query + self.user_embeddings(user_id).unsqueeze(1)
        tokens = torch.cat([clean_history, query], dim=1)
        timesteps = torch.zeros(history.shape[0], device=history.device, dtype=torch.long)
        predicted = self.denoise(tokens, history_mask, text_cond, image_cond, timesteps, clean_history)
        logits = self.compute_logits(predicted[:, -1, :])
        logits[:, 0] = -1e9
        return logits

    @staticmethod
    def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded_mask = mask.unsqueeze(-1).to(prediction.dtype)
        squared_error = ((prediction - target) ** 2) * expanded_mask
        return squared_error.sum() / expanded_mask.sum().clamp(min=1.0)

    def _build_sampling_schedule(self, inference_steps: int | None) -> list[int]:
        total_steps = self.scheduler.num_steps
        if inference_steps is None or inference_steps >= total_steps:
            return list(range(total_steps - 1, -1, -1))

        schedule = torch.linspace(0, total_steps - 1, steps=inference_steps, device=self.position_embeddings.device)
        unique_steps = sorted({int(round(step.item())) for step in schedule}, reverse=True)
        if unique_steps[-1] != 0:
            unique_steps.append(0)
        return unique_steps

    def forward(
        self,
        user_id: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        target: torch.Tensor,
        text_cond: torch.Tensor,
        image_cond: torch.Tensor,
    ) -> DitForRecOutput:
        clean_tokens = self.build_clean_tokens(user_id, history, target)
        clean_history = clean_tokens[:, :-1, :]
        timesteps = torch.randint(0, self.scheduler.num_steps, (history.shape[0],), device=history.device)
        noise = torch.randn_like(clean_tokens)
        noisy_tokens = self.scheduler.q_sample(clean_tokens, timesteps, noise)
        pred_clean = self.denoise(noisy_tokens, history_mask, text_cond, image_cond, timesteps, clean_history)

        pred_target = pred_clean[:, -1, :]
        gold_target = clean_tokens[:, -1, :]

        logits = self.compute_logits(pred_target)
        logits[:, 0] = -1e9
        direct_logits = self.direct_score_logits(user_id, history, history_mask, text_cond, image_cond)

        token_mask = torch.cat([history_mask, torch.ones_like(history_mask[:, :1])], dim=1)
        denoise_loss = self._masked_mse(pred_clean, clean_tokens, token_mask)
        target_recon_loss = F.mse_loss(pred_target, gold_target)
        prior_loss = self.scheduler.prior_matching_loss(clean_tokens)
        ce_loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        direct_ce_loss = F.cross_entropy(direct_logits, target, label_smoothing=self.label_smoothing)
        loss = (
            self.denoise_weight * denoise_loss
            + self.target_recon_weight * target_recon_loss
            + self.prior_weight * prior_loss
            + self.ce_weight * ce_loss
            + self.direct_ce_weight * direct_ce_loss
        )

        return DitForRecOutput(
            loss=loss,
            denoise_loss=denoise_loss,
            target_recon_loss=target_recon_loss,
            prior_loss=prior_loss,
            ce_loss=ce_loss,
            direct_ce_loss=direct_ce_loss,
            logits=logits,
            pred_target=pred_target,
        )

    @torch.no_grad()
    def sample_logits(
        self,
        user_id: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        text_cond: torch.Tensor,
        image_cond: torch.Tensor,
        inference_steps: int | None = None,
        sampling_strategy: str = "ddim",
        eta: float = 0.0,
        noise_history: bool = True,
    ) -> torch.Tensor:
        if self.direct_score_weight >= 1.0:
            return self.direct_score_logits(user_id, history, history_mask, text_cond, image_cond)

        device = history.device
        schedule = self._build_sampling_schedule(inference_steps)

        clean_history = self.build_history_tokens(user_id, history)
        history_noise = torch.randn_like(clean_history) if noise_history else torch.zeros_like(clean_history)
        current_target = torch.randn(history.shape[0], 1, self.hidden_dim, device=device)

        for index, step in enumerate(schedule):
            timesteps = torch.full((history.shape[0],), step, device=device, dtype=torch.long)
            noisy_history = (
                self.scheduler.q_sample(clean_history, timesteps, history_noise) if noise_history else clean_history
            )
            current_tokens = torch.cat([noisy_history, current_target], dim=1)
            pred_clean = self.denoise(current_tokens, history_mask, text_cond, image_cond, timesteps, clean_history)
            pred_target = pred_clean[:, -1:, :]

            if index == len(schedule) - 1:
                current_target = pred_target
                break

            prev_step = schedule[index + 1]
            prev_timesteps = torch.full((history.shape[0],), prev_step, device=device, dtype=torch.long)
            if sampling_strategy.lower() == "ddpm" and prev_step == step - 1:
                current_target = self.scheduler.posterior_sample(current_target, pred_target, timesteps)
            else:
                current_target = self.scheduler.ddim_step(current_target, pred_target, timesteps, prev_timesteps, eta=eta)

        logits = self.compute_logits(current_target[:, 0, :])
        logits[:, 0] = -1e9
        if self.direct_score_weight > 0.0:
            direct_logits = self.direct_score_logits(user_id, history, history_mask, text_cond, image_cond)
            logits = (1.0 - self.direct_score_weight) * logits + self.direct_score_weight * direct_logits
            logits[:, 0] = -1e9
        return logits
