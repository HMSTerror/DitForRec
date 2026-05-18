# DitForRec

DiT-style sequential recommendation project for Amazon Beauty / Toys (2014) with:

- paper-style `5-core` preprocessing
- maximum history length `50`
- `leave-last-two` split: second-last for validation, last for test
- lookup-table item/history embeddings as the main DiT input
- T5 text conditioning and SigLIP image conditioning
- hierarchical cross-attention injection: text layer + image layer
- DCRec-inspired explicit correction with clean history and timestep inside blocks and after the last block

The implementation is designed to be practical rather than perfectly paper-faithful. It keeps the project runnable on a toy dataset while preserving the core modeling ideas requested for Amazon Beauty.

The current codebase now includes a more paper-aligned diffusion objective and reverse process:

- full-sequence denoising loss over history + target tokens
- target reconstruction loss for retrieval quality stabilization
- prior matching regularization from the terminal diffusion state to the Gaussian prior
- domain-transition style retrieval cross-entropy over the item table
- DDPM/DDIM-compatible reverse sampling with configurable few-step inference

## References

This project is primarily inspired by:

- DiT: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- DCRec: [Dual Conditional Diffusion Models for Sequential Recommendation](https://arxiv.org/abs/2410.21967)
- SASRec: [Self-Attentive Sequential Recommendation](https://openreview.net/forum?id=Hkg0u3Etwr)
- DiffuRec: [Sequential Recommendation with Diffusion Models](https://arxiv.org/abs/2306.12514)
- DreamRec: [DreamRec: Sequential Recommendation with Diffusion Probabilistic Models](https://arxiv.org/abs/2306.10103)
- BERT4Rec: [Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)

## Project Layout

```text
DitForRec/
  configs/
  scripts/
  src/ditforrec/
    data/
    model/
```

## Environment

Recommended Python: `3.10` to `3.11`

```powershell
cd C:\Users\14466\Documents\Playground\DitForRec
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Build the toy dataset

```powershell
python -m ditforrec.data.toy --output-root data
python -m ditforrec.data.preprocess --dataset toy --data-root data --extract-text --extract-image --text-backbone google-t5/t5-small --image-backbone google/siglip-base-patch16-224
```

### 2. Train on toy

```powershell
python -m ditforrec.train --config configs/toy_debug.yaml
```

### 3. Evaluate on toy

```powershell
python -m ditforrec.evaluate --config configs/toy_debug.yaml --checkpoint outputs/toy_debug/best.pt
```

## Amazon Beauty / Toys 2014

### Download + preprocess

Beauty:

```powershell
python -m ditforrec.data.preprocess --dataset beauty --data-root data --download --extract-text --extract-image
```

Toys:

```powershell
python -m ditforrec.data.preprocess --dataset toys --data-root data --download --extract-text --extract-image
```

The downloader targets the Amazon 2014 files commonly used by classical sequential recommendation work:

- `reviews_Beauty_5.json.gz`
- `meta_Beauty.json.gz`
- `reviews_Toys_and_Games_5.json.gz`
- `meta_Toys_and_Games.json.gz`

## Preprocessing Protocol

The preprocessing follows common sequential recommendation practice used by SASRec/DCRec-style pipelines:

1. Load Amazon 2014 5-core review interactions.
2. Keep only valid `(user, item, timestamp)` events.
3. Iteratively apply `5-core` filtering so all remaining users and items have at least 5 interactions.
4. Sort each user sequence by timestamp.
5. Truncate each sequence to the latest `50` interactions.
6. Split per user:
   - train: all but the last two interactions
   - validation: the second-last interaction
   - test: the last interaction
7. Build prefix-target training instances from the train part.
8. Pad histories with item id `0`.

Output files live under:

```text
data/processed/<dataset>/
  mappings.json
  item_metadata.jsonl
  train.jsonl
  val.jsonl
  test.jsonl
  features/
    item_text.npy
    item_image.npy
    feature_manifest.json
```

## Model Summary

Main sequence input:

- lookup-table history item embeddings
- lookup-table target item embedding
- optional user embedding broadcast to sequence tokens
- Gaussian noising on the concatenated sequence, following diffusion SR conventions

Auxiliary conditioning:

- history-side text features from T5 encoder
- history-side image features from SigLIP vision encoder
- text cross-attention injected in one configurable block
- image cross-attention injected in another configurable block

Explicit correction:

- DCRec-inspired conditional layer norm using clean history + timestep
- explicit clean-history cross-attention inside each DiT block
- final clean-history correction applied after the last block

Prediction:

- recover target item embedding
- compare against the item embedding table by cosine similarity
- optimize denoising + target reconstruction + prior matching + retrieval cross-entropy

## Training / Evaluation Outputs

Training now writes RecBole-style artifacts under `outputs/<experiment_name>/`:

- `train.log`: epoch-level and step-level logs
- `best.pt`: best checkpoint selected by the configured validation metric
- `best_metrics.json`: best validation/test summary
- `train_history.jsonl`: structured epoch history
- `config_snapshot.yaml`: copied runtime config

Evaluation writes:

- `eval.log`
- `eval_metrics.json`

The default evaluation is full-sort with historical-item masking and reports:

- `Hit@K`
- `Recall@K`
- `NDCG@K`
- `MRR@K`
- `Precision@K`

## Ablation Configs

Example server-ready ablation configs live under `configs/ablations/`:

- `beauty_no_multimodal.yaml`
- `beauty_no_correction.yaml`
- `beauty_id_only_ddpm.yaml`
- `beauty_few_step_ddim.yaml`

## Important Notes

- The text/image encoders require Hugging Face downloads on first use.
- Amazon 2014 metadata may contain missing image URLs. The pipeline tolerates missing image features by zero-filling and masking.
- The current implementation favors clarity and extensibility over benchmark tuning.
- The toy dataset is the recommended smoke test because it is generated locally and does not depend on external downloads.

## Suggested Commands

See [scripts/bootstrap.ps1](C:\Users\14466\Documents\Playground\DitForRec\scripts\bootstrap.ps1) for an end-to-end bootstrap example.
