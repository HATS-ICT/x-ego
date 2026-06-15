# Training & Testing Pipeline

How `main.py` dispatches into training/testing, how the Lightning `Trainer` is configured, and how checkpointing/resume work.

Relevant code: `src/train/run_tasks.py`, `src/train/train_pipeline.py`, `src/train/test_pipeline.py`, `src/train/training_utils.py`.

## Dispatch

`main.py` calls one of four thin wrappers in `run_tasks.py`, each pairing a model class with a datamodule and delegating to a generic pipeline:

| Function | Model | DataModule | Pipeline |
|----------|-------|------------|----------|
| `train_contrastive` | `ContrastiveModel` | `ContrastiveDataModule` | `run_training_pipeline` |
| `test_contrastive` | `ContrastiveModel` | `ContrastiveDataModule` | `run_test_only_pipeline` |
| `train_downstream` | `LinearProbeModel` | `DownstreamDataModule` | `run_training_pipeline` |
| `test_downstream` | `LinearProbeModel` | `DownstreamDataModule` | `run_test_only_pipeline` |

## Training pipeline (`run_training_pipeline`)

Steps (`train_pipeline.py`):

1. `torch.set_float32_matmul_precision('medium')`; `L.seed_everything(cfg.meta.seed, workers=True)`.
2. **Data**: instantiate the datamodule, `prepare_data()`, `setup("fit")`; print train/val counts.
3. **Model**: instantiate the model; optionally `torch.compile` the encoder (`compile_video_encoder_if_requested`).
4. **Callbacks & logger**: `setup_callbacks(cfg)`, `setup_logger(cfg)`.
5. **Trainer**: `create_trainer(cfg, callbacks, logger)`; log hyperparams to wandb if enabled.
6. **Fit**: `trainer.fit(...)` — resumes from `cfg.checkpoint.resume_checkpoint_path` if present.
7. **Test after training**: `setup("test")`; test the `last` checkpoint, plus `best` if a `ModelCheckpoint` with `save_top_k > 0` recorded one. Each checkpoint is tested with a fresh model instance.
8. `wandb.finish()` if logging.

### `torch.compile`

`compile_video_encoder_if_requested` only fires when `cfg.training.torch_compile` is true **and** the platform is Linux. It compiles `model.video_encoder` (Lightning, logging, and metrics stay eager). Already-compiled modules are skipped.

## Test-only pipeline (`run_test_only_pipeline`)

Used by `--mode test`. Sets seed/precision, sets up the datamodule for `"test"` (returns early if there is no test split), and tests checkpoints found via `find_saved_checkpoints(cfg.path.ckpt)` — `last.ckpt` and/or `best.ckpt` (errors if neither exists), or a single explicitly-passed checkpoint path. A logger is created only if `cfg.wandb.enabled`.

## Trainer configuration

`create_trainer` (`train_pipeline.py`) passes all of `cfg.training` to `L.Trainer`: `max_epochs`, `max_steps`, `accelerator`, `devices`, `strategy`, `precision`, `accumulate_grad_batches`, `val_check_interval`, `check_val_every_n_epoch`, `log_every_n_steps`, the `enable_*` flags, `deterministic`, `num_sanity_val_steps`, and `limit_{train,val,test}_batches`.

**Gradient clipping** is handled carefully:
- With contrastive manual optimization (`contrastive_accumulate_batches > 1`) or a fused AdamW optimizer, Lightning's trainer-level `gradient_clip_val` is set to `None` (those paths clip manually / can't clip at the trainer level).
- Otherwise it uses `cfg.training.gradient_clip_val`.

`create_test_trainer` (`test_pipeline.py`) is minimal: accelerator, devices, precision, logger, progress bar, `limit_test_batches`, no callbacks.

## Callbacks & logging (`training_utils.py`)

- **`setup_callbacks`** builds `ModelCheckpoint`(s) from `cfg.checkpoint`. If both `epoch` and `step` sub-blocks exist, it creates two checkpoint callbacks (one per-epoch, one per-step) with their own `monitor`/`mode`/`save_top_k`/`save_last`. Adds `EarlyStopping` if `cfg.early_stopping` is present, and always a `ModelSummary(max_depth=2)`.
- **`setup_logger`** returns `None` unless `cfg.wandb.enabled` and `cfg.wandb.save_dir` are set. It auto-derives tags from `meta.run_name` and the contrastive setting (`yes_contra`/`no_contra`) and creates a `WandbLogger`.

## Resume and test-only

| | `--mode test` | resume training (`meta.resume_exp` in train/dev) |
|---|---|---|
| Config source | saved `hparam.yaml` (config files skipped) | current config files + saved paths |
| What loads | checkpoint via `find_saved_checkpoints` | `find_resume_checkpoint` (`last.ckpt` → newest `*.ckpt`) |
| Effect | runs `trainer.test` only | `trainer.fit(..., ckpt_path=...)` continues training |

Both go through `setup_directory`/`setup_resume_cfg` in `main.py`, which point `path.exp/ckpt/plots` at the existing experiment directory rather than creating a new one. See [configuration.md](configuration.md#experiment-directories).

## Metrics & plots

- **`src/utils/metric_utils.py`** — histogram/count metrics for location tasks: `kl_divergence_histogram`, `multinomial_loss`, `exact_match_accuracy`, `l1_count_error`, `chamfer_distance_batch`. (Per-task `self.log(...)` calls live in the model classes.)
- **`src/utils/plot_utils.py`** — uses `awpy` map backgrounds to render prediction scatter plots, count distributions, and KDE heatmaps of predicted vs ground-truth agent locations (`create_prediction_plots`, `create_regression_plots`, `create_classification_plots`, `create_prediction_heatmaps[_grid]`). Outputs land in the experiment's `plots/` dir.
