from __future__ import annotations

from ditforrec.data.dataset import SequentialRecommendationDataset
from ditforrec.model.ditforrec import DitForRec


def build_model(config, dataset: SequentialRecommendationDataset) -> DitForRec:
    model_cfg = config.model
    training_cfg = config.training
    text_dim = int(getattr(dataset.item_text, "shape", [0, model_cfg.text_dim])[1]) if dataset.item_text.ndim == 2 else int(model_cfg.text_dim)
    image_dim = int(getattr(dataset.item_image, "shape", [0, model_cfg.image_dim])[1]) if dataset.item_image.ndim == 2 else int(model_cfg.image_dim)
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
        text_dim=text_dim,
        image_dim=image_dim,
        text_inject_layers=list(model_cfg.get("text_inject_layers", [])),
        image_inject_layers=list(model_cfg.get("image_inject_layers", [])),
        timestep_dim=model_cfg.timestep_dim,
        use_user_embeddings=bool(model_cfg.get("use_user_embeddings", True)),
        use_text_condition=bool(model_cfg.get("use_text_condition", True)),
        use_image_condition=bool(model_cfg.get("use_image_condition", True)),
        use_history_correction=bool(model_cfg.get("use_history_correction", True)),
        use_final_correction=bool(model_cfg.get("use_final_correction", True)),
        add_user_to_target=bool(model_cfg.get("add_user_to_target", True)),
        diffusion_beta_start=float(model_cfg.get("diffusion_beta_start", 1e-4)),
        diffusion_beta_end=float(model_cfg.get("diffusion_beta_end", 2e-2)),
        denoise_weight=float(training_cfg.get("denoise_weight", training_cfg.get("sequence_weight", 1.0))),
        target_recon_weight=float(training_cfg.get("target_recon_weight", training_cfg.get("recon_weight", 1.0))),
        prior_weight=float(training_cfg.get("prior_weight", 0.0)),
        ce_weight=float(training_cfg.get("ce_weight", 1.0)),
        direct_ce_weight=float(training_cfg.get("direct_ce_weight", 0.0)),
        direct_score_weight=float(training_cfg.get("direct_score_weight", 0.0)),
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        logit_temperature=float(training_cfg.get("logit_temperature", 1.0)),
    )
