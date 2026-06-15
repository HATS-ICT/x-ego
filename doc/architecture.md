# Model Architecture

This doc covers the two LightningModules and the shared encoder zoo.

Relevant code: `src/models/contrastive_model.py`, `src/models/downstream.py`, `src/models/modules/`.

## The encoder zoo (`src/models/modules/`)

A single factory, `VideoEncoder` (`video_encoder.py`), selects a backbone by `model_type` and exposes a uniform interface:

```python
encoder = VideoEncoder(cfg.model.encoder)
embeddings = encoder(pixel_values)              # [B, embed_dim]
encoder.embed_dim                               # output dimensionality
encoder(pixel_values, return_temporal_features=True)   # per-frame/temporal features
```

### Supported backbones

`model_type` accepts an alias **or** the full HuggingFace name (`normalize_model_type` resolves both). Aliases → backbones:

| Alias | Backbone | Type | Pooling / notes |
|-------|----------|------|-----------------|
| `clip` | `openai/clip-vit-base-patch32` | image | mean over patch tokens; `embed_dim = hidden_size` |
| `siglip2` | `google/siglip2-base-patch16-224` | image | mean over patches (no CLS) |
| `dinov2` | `facebook/dinov2-base` | image | **CLS ⊕ mean(patches)** → `embed_dim = hidden_size × 2` |
| `dinov3` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | image | CLS ⊕ mean(patches) → `hidden_size × 2` |
| `resnet50` | torchvision ResNet-50 (**from scratch**, no pretrained weights) | image | `embed_dim = 2048`; `finetune_last_k_layers` ignored |
| `vivit` | `google/vivit-b-16x2-kinetics400` | native video | resamples to 32 frames; returns CLS |
| `videomae` | `MCG-NJU/videomae-base` | native video | resamples to 16 frames; mean-pool or CLS |
| `vjepa2` | `facebook/vjepa2-vitl-fpc16-256-ssv2` | native video | `VJEPA2AttentivePooler` |

**Image encoders** process `[B, T, C, H, W]` by flattening to `[B·T, C, H, W]`, encoding each frame, then mean-pooling over time. **Native video encoders** consume the temporal dimension directly and resample to their expected frame count via `temporal_sampling` (trilinear interpolation along time).

### Temporal transformer (image encoders only)

For `clip`/`siglip2`/`dinov2`/`dinov3`, setting `model.encoder.temporal_depth > 0` inserts a `TemporalTransformer` (`architecture_utils.py`) that does cross-frame attention on patch tokens `[B,T,P,E]` *before* spatial pooling. `temporal_heads` must evenly divide the hidden size. Native video models ignore these (they model time internally).

### Finetuning control

`configure_finetuning(model, encoder_layers, finetune_last_k_layers, final_norm)`:
- `-1` → finetune all layers
- `0` → freeze entirely
- `k` → unfreeze the last *k* transformer layers plus the final norm

`get_embed_dim_for_model_type` returns the output dim from the HF config without loading weights (handy for inspection — see `check_model_param.py`).

### Other module files

| File | Contents |
|------|----------|
| `architecture_utils.py` | `build_mlp` (the projector), activation map, `TemporalAttention`, `SwiGLUFFN`, `TemporalTransformerBlock`, `TemporalTransformer`. |
| `norm.py` | `RMSNorm`, `SimpleLayerNorm`, `AdaptiveNormalizer` (FiLM conditioning). |
| `optimizer_utils.py` | `build_optimizer` supporting `adamw` (optionally fused) and `muon` (with a single-process Muon-with-aux-AdamW variant). |

## Stage 1 — `ContrastiveModel`

`src/models/contrastive_model.py`. Pipeline: `VideoEncoder → video_projector (MLP) → SigLIP-style sigmoid loss`.

### Variable-agent batching (no padding)

Dead agents are never padded. A batch packs all alive agents into the first dimension:

```
video        : [total_agents, T, C, H, W]
agent_counts : [B]   # how many agents belong to each sample; sums to total_agents
```

e.g. `agent_counts=[3,2]` → `video` has 5 rows: the first 3 are sample 0's teammates, the next 2 are sample 1's.

### Objective

`create_alignment_matrix(agent_counts)` builds an `[N,N]` block-diagonal label matrix: `1` where two agents share a sample (same team, same game state, different viewpoint), `0` otherwise. `compute_contrastive_loss` then applies the SigLIP sigmoid loss with learnable `logit_scale` and `logit_bias`:

```python
logits = (normalize(emb) @ normalize(emb).T) * logit_scale.exp() + logit_bias
loglik = logsigmoid((2*labels - 1) * logits)   # diagonal (self-similarity) masked out
loss   = -loglik.sum(-1).mean() * loss_weight
```

Logged metrics include `binary_acc`, `pos_pair_acc`, `neg_pair_acc`, `temperature`, and retrieval `top1/3/5_acc`.

### Embedding-cache gradient accumulation

Large contrastive batches don't fit in memory, so `contrastive_accumulate_batches > 1` switches Lightning to **manual optimization**. Microbatches are run forward and their projected embeddings (detached) are cached along with RNG state; once `contrastive_accumulate_batches` microbatches are collected, a single large loss is computed and backpropped to the cached projections, then each microbatch is **replayed** (RNG restored) to push gradients into the encoder. This yields a large effective contrastive batch at small memory cost. (`find_max_contra_accumulate_batches.py` probes the safe limits per encoder.)

### Stage-1 → Stage-2 handoff

- `get_encoder_state_dict()` returns the `video_encoder` (+ projector, logit params).
- `load_encoder_from_checkpoint(path, cfg)` reconstructs `(video_encoder, video_projector)` for reuse.
- `on_load_checkpoint`/`_strip_orig_mod_prefix` strip the `_orig_mod.` prefix added by `torch.compile` so compiled and uncompiled checkpoints are interchangeable.

## Stage 2 — `LinearProbeModel`

`src/models/downstream.py`. Pipeline: `VideoEncoder → LinearProbeHead → task prediction`.

### Encoder init

If `cfg.model.stage1_checkpoint` is set, only the `video_encoder.*` weights from the Stage 1 checkpoint are loaded (the projector is discarded). If it's null, the off-the-shelf HF encoder is used as a **baseline**. With `encoder.trainable=false` (default), the encoder is frozen and run under `torch.no_grad()`; only the head trains.

### Heads by `ml_form`

`LinearProbeHead` is a single `nn.Linear` whose shape and loss depend on the task's `ml_form`:

| `ml_form` | Head out | Loss | Metrics |
|-----------|----------|------|---------|
| `binary_cls` | 1 | BCE-with-logits | accuracy, F1, AUROC |
| `multi_cls` | `num_classes` | cross-entropy | accuracy (+top3/top5), macro-F1 |
| `multi_label_cls` | `num_classes` | BCE-with-logits | exact-match, Hamming, F1, AUROC |
| `regression` | `output_dim` | MSE | MSE, MAE, R² |

`on_test_epoch_end` writes `test_results_<checkpoint_name>.json` into the experiment dir with task metadata and metrics. See [tasks.md](tasks.md) for the list of tasks and [training.md](training.md) for how testing is driven.
