from __future__ import annotations

from ditforrec.data.dataset import SequentialRecommendationDataset
from ditforrec.model.ditforrec import DitForRec


def build_model(config, dataset: SequentialRecommendationDataset) -> DitForRec:
    model_cfg = config.model
    training_cfg = config.training
    return DitForRec(
        num_items=dataset.num_items,
        num_users=dataset.num_users,
        hidden_dim=model_cfg.hidden_dim,
        num_heads=model_cfg.num_heads,
        depth=model_cfg.depth,
        mlp_ratio=model_cfg.mlp_ratio,
        dropout=model_cfg.dropout,
        max_history=model_cfg.max_history,
        num_diffusion_steps=model_cfg.num_diffusion_steps,
        text_dim=model_cfg.text_dim,
        image_dim=model_cfg.image_dim,
        text_inject_layers=list(model_cfg.get("text_inject_layers", [])),
        image_inject_layers=list(model_cfg.get("image_inject_layers", [])),
        timestep_dim=model_cfg.timestep_dim,
        use_user_embeddings=bool(model_cfg.get("use_user_embeddings", True)),
        use_text_condition=bool(model_cfg.get("use_text_condition", True)),
        use_image_condition=bool(model_cfg.get("use_image_condition", True)),
        use_history_correction=bool(model_cfg.get("use_history_correction", True)),
        use_final_correction=bool(model_cfg.get("use_final_correction", True)),
        denoise_weight=float(training_cfg.get("denoise_weight", training_cfg.get("sequence_weight", 1.0))),
        target_recon_weight=float(training_cfg.get("target_recon_weight", training_cfg.get("recon_weight", 1.0))),
        prior_weight=float(training_cfg.get("prior_weight", 0.0)),
        ce_weight=float(training_cfg.get("ce_weight", 1.0)),
        logit_temperature=float(training_cfg.get("logit_temperature", 1.0)),
    )
