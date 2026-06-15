# Configuration

X-Ego uses [OmegaConf](https://omegaconf.readthedocs.io/) YAML configs with full CLI override support. This doc covers the config files, override syntax, environment variables, and the experiment-directory convention.

Relevant code: `src/utils/config_utils.py`, `src/utils/env_utils.py`, `src/utils/experiment_utils.py`, and `configs/`.

## Config files

```
configs/
├── train/
│   ├── contrastive.yaml      # Stage 1 full-training config
│   └── downstream.yaml       # Stage 2 full-training config
└── dev/
    ├── contrastive.yaml      # tiny overrides for smoke tests
    └── downstream.yaml
```

The `train/<task>.yaml` files are the **complete** configs. The `dev/<task>.yaml` files are **partial overrides** merged on top in `dev` mode (smaller batches, 2 epochs, a few limited batches, wandb disabled).

### Priority order

Each level overrides the one below it (right operand wins in `OmegaConf.merge`):

```
CLI overrides  >  configs/dev/<task>.yaml (dev mode)  >  configs/train/<task>.yaml
```

In `test` mode, config files are skipped entirely; the saved `hparam.yaml` is the base, with CLI overrides applied on top.

## CLI override syntax

Pass trailing `key.subkey=value` tokens after the flags. They are parsed via `OmegaConf.from_dotlist`:

```bash
python main.py --mode train --task downstream \
    task.task_id=enemy_location_5s \
    model.encoder.model_type=clip \
    data.batch_size=8 \
    data.random_mask.enable=false \
    meta.seed=123
```

> **Overrides cannot introduce new keys.** `apply_cfg_overrides` raises `KeyError` if the target key does not already exist in the loaded YAML — this catches typos. Add the setting to the YAML first if you genuinely need a new key.

## Top-level config sections

Both configs share most sections; `downstream` adds a `task` block and drops the contrastive-only pieces.

| Section | Purpose |
|---------|---------|
| `meta` | `seed`, `run_name` (interpolated, e.g. `contrastive-${model.encoder.model_type}`), `resume_exp`. |
| `data` | dataset paths, batching, video loading, masking — see below. |
| `model` | encoder + (Stage 1) projector/contrastive head, or (Stage 2) `stage1_checkpoint` + probe settings. |
| `training` | Lightning `Trainer` args (`max_epochs`, `precision`, `devices`, `accumulate_grad_batches`, …). |
| `optimization` | optimizer (`muon` for Stage 1, `adamw` for Stage 2), LR, scheduler. |
| `wandb` | logging config; `enabled: false` disables. |
| `checkpoint` | `ModelCheckpoint` settings (`epoch` and `step` sub-blocks). |
| `task` (downstream only) | `task_id`, `ml_form`, `num_classes`, `output_dim`, `label_column`. |

### Notable `model.encoder` fields

| Field | Meaning |
|-------|---------|
| `model_type` | encoder alias or full HF name (see [architecture.md](architecture.md)). |
| `finetune_last_k_layers` | `-1` = all, `0` = freeze, `k` = unfreeze last *k* layers + final norm. |
| `temporal_heads`, `temporal_depth` | enable a temporal transformer over frames (image encoders only). `temporal_heads` must divide the hidden size. |
| `trainable` (downstream) | `false` = linear probe (frozen encoder); `true` = train encoder + head. |

### Notable `data` fields

| Field | Meaning |
|-------|---------|
| `map` | which map's data subtree to use (`inferno`, `dust2`, `mirage`). |
| `video_folder` | processed clips dir (default `video_306x306_4fps`). |
| `batch_size` | for contrastive this is a **total-video budget** consumed by `FixedVideoBatchSampler`, not a sample count (see [data-pipeline.md](data-pipeline.md)). |
| `fixed_duration_seconds`, `target_fps` | clip length and sampled FPS → `T = duration × fps` frames. |
| `ui_mask` | `none` / `minimap_only` / `all` — mask out HUD regions. |
| `random_mask.*` | random tube masking for augmentation (Stage 1 only by default). |
| `labels_folder`, `labels_filename` | resolve the label CSV; downstream uses `labels/all_tasks` + `${task.task_id}.csv`. |

> The contrastive config is validated by `validate_contrastive_cfg`, which enforces ~90 required keys plus semantic checks (e.g. `accumulate_grad_batches` must be `1`, `ui_mask` must be valid, `contrastive.enable` must be true).

## Downstream task auto-configuration

For `--task downstream`, `main.py` calls `apply_task_config(cfg, data_path)`. This reads `cfg.task.task_id` + `cfg.data.map`, looks up `task_definitions.csv`, and fills in `task.ml_form`, `task.num_classes`, `task.output_dim`, `task.label_column`, and `data.labels_filename`. Location/place tasks get a map-specific `output_dim` (the number of named places on that map). If the CSV can't be found it falls back to the YAML values. Details in [tasks.md](tasks.md).

## Environment variables

`src/utils/env_utils.py` calls `load_dotenv()` at import, so a `.env` file in the repo root is read automatically. Three variables are required — **there are no fallback defaults** (an unset var returns `None`):

| Variable | Used by | Meaning |
|----------|---------|---------|
| `SRC_BASE_PATH` | `get_src_base_path()` | path to this repository. |
| `DATA_BASE_PATH` | `get_data_base_path()` | dataset root (per-map subtrees live here). |
| `OUTPUT_BASE_PATH` | `get_output_base_path()` | where experiment directories are written. |

These are injected into the config as `path.src`, `path.data`, `path.output` by `setup_base_pathing` in `main.py`. Run `python -m src.utils.env_utils` to print the resolved values.

## Experiment directories

`src/utils/experiment_utils.py` manages output. A fresh run creates:

```
<OUTPUT_BASE_PATH>/<run_name>-<YYMMDD-HHMMSS>-<hash4>/
├── hparam.yaml             # the full resolved config (OmegaConf.save)
├── checkpoint/             # *.ckpt, last.ckpt, best.ckpt
├── plots/
└── evaluation_results/     # created on demand: <type>_eval_<timestamp>.json
```

- `run_name` comes from `meta.run_name` (e.g. `probe-self_location_0s-siglip2`); the timestamp uses a 2-digit year, and `hash4` is 4 random alphanumerics.
- `save_hyperparameters` writes `hparam.yaml`; `load_experiment_cfg` reads it back (used by `--mode test`).

### Resume

When `meta.resume_exp=<exp_name>` is set, `setup_resume_cfg` locates the existing dir and its checkpoint (`find_resume_checkpoint` prefers `last.ckpt`, else the most recent `*.ckpt`), and wires `checkpoint.resume_checkpoint_path` so training continues from it. See [training.md](training.md#resume-and-test-only).
