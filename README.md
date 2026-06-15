# X-Ego: Cross-Egocentric Multi-Agent Video Understanding in Counter-Strike 2

[![Paper](https://img.shields.io/badge/arXiv-2510.19150-b31b1b.svg)](https://arxiv.org/abs/2510.19150)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue.svg)](https://huggingface.co/datasets/wangyz1999/X-EGO-CS)

**X-Ego-CS** is a large-scale multi-agent gameplay dataset for **cross-egocentric video understanding** in *Counter-Strike 2*.
It provides synchronized **first-person (egocentric)** video streams from **10 players (5v5)** across professional-level matches, enabling research on **tactical reasoning**, **situational awareness**, and **multi-agent coordination**.

https://github.com/user-attachments/assets/5485943e-95e4-4f0d-ae16-4cde4744bd2f

![x-ego](https://github.com/user-attachments/assets/c81e440f-3246-493e-9354-0a845157c581)

---

## What this repository does

X-Ego trains and evaluates video encoders with a **two-stage pipeline**:

1. **Stage 1 — Cross-Egocentric Contrastive Learning (`contrastive`).** A video encoder is trained so that synchronized clips from players *on the same team* (the same game state seen from different first-person viewpoints) produce similar embeddings. This is a SigLIP-style sigmoid contrastive objective over a variable number of alive teammates per sample.
2. **Stage 2 — Downstream Probing (`downstream`).** The Stage 1 encoder (or an off-the-shelf encoder, as a baseline) is frozen and a lightweight linear head is trained on one of ~39 tactical prediction tasks (player/teammate/enemy location, alive counts, combat events, bomb state, round outcome, movement, speed, …).

`main.py` is the single entry point for both stages. Everything else in the repository is either a data-preparation script, an experiment driver, or an analysis/visualization tool.

> **New to the code? Start with [`doc/overview.md`](doc/overview.md)**, then dive into the focused docs linked at the bottom of this README.

---

## Quick Start

We use [`uv`](https://docs.astral.sh/uv) for package management. After installing `uv`, run:

```bash
uv sync
```

to create the virtual environment and install all dependencies (PyTorch CUDA 12.8 wheels are used on Linux/Windows).

### 1. Configure paths

The code reads three environment variables (loaded from a `.env` file in the repo root via `python-dotenv`):

```bash
# .env
SRC_BASE_PATH=/path/to/x-ego          # this repository
DATA_BASE_PATH=/path/to/data          # dataset root (see "Data Layout" below)
OUTPUT_BASE_PATH=/path/to/output      # where experiment dirs are written
```

These have **no defaults** — they must be set before training. See [`doc/configuration.md`](doc/configuration.md#environment-variables).

### 2. Train

```bash
# Stage 1: contrastive pre-training
python main.py --mode train --task contrastive

# Stage 2: downstream probe with an off-the-shelf encoder (baseline)
python main.py --mode train --task downstream task.task_id=self_location_0s

# Stage 2: downstream probe initialized from a Stage 1 checkpoint
python main.py --mode train --task downstream \
    task.task_id=self_location_0s \
    model.stage1_checkpoint=/path/to/contrastive/checkpoint.ckpt
```

### 3. Modes

| `--mode` | Behavior |
|----------|----------|
| `train`  | Full training, then testing on the best/last checkpoint. |
| `dev`    | Same as `train` but merges `configs/dev/<task>.yaml` on top — tiny batches, 2 epochs, a few steps, wandb off. Use for smoke tests. |
| `test`   | Test-only. **Skips config files** and loads `hparam.yaml` from a saved experiment. Requires `meta.resume_exp=<exp_name>`. |

```bash
# Test an existing experiment
python main.py --mode test --task downstream \
    meta.resume_exp=probe-self_location_0s-siglip2-260120-152532-pe8y
```

### 4. Override any config from the CLI

Every key in the YAML configs can be overridden with `key.subkey=value`:

```bash
python main.py --mode train --task contrastive \
    training.max_epochs=20 data.batch_size=8 meta.seed=123 \
    model.encoder.model_type=dinov3
```

Configuration priority (higher overrides lower):

```
CLI overrides  >  configs/dev/<task>.yaml (dev mode only)  >  configs/train/<task>.yaml
```

See [`doc/configuration.md`](doc/configuration.md) for the full config reference.

---

## Supported encoders

Set via `model.encoder.model_type`. Aliases map to HuggingFace / torchvision backbones:

| Alias | Backbone | Family |
|-------|----------|--------|
| `clip` | `openai/clip-vit-base-patch32` | image (per-frame) |
| `siglip2` | `google/siglip2-base-patch16-224` | image (per-frame) |
| `dinov2` | `facebook/dinov2-base` | image (per-frame) |
| `dinov3` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | image (per-frame) |
| `vivit` | `google/vivit-b-16x2-kinetics400` | native video |
| `videomae` | `MCG-NJU/videomae-base` | native video |
| `vjepa2` | `facebook/vjepa2-vitl-fpc16-256-ssv2` | native video |
| `resnet50` | torchvision ResNet-50 (from scratch) | image (per-frame) |

Image encoders optionally add a temporal transformer over frames (`model.encoder.temporal_depth`/`temporal_heads`). See [`doc/architecture.md`](doc/architecture.md).

---

## Data Layout

`DATA_BASE_PATH` points at a per-map dataset root. Paths the loaders expect:

```
<DATA_BASE_PATH>/
└── <map>/                                  # e.g. inferno, dust2, mirage
    ├── time_offset.json                    # video↔game-time sync per match/agent/round
    ├── video_306x306_4fps/                 # processed clips (data.video_folder)
    │   └── <match_id>/<player_id>/round_<nn>.mp4
    └── labels/
        ├── contrastive.csv                 # Stage 1 pairs (no labels)
        ├── task_definitions.csv            # task registry (ml_form, num_classes, …)
        └── all_tasks/
            └── <task_id>.csv               # one label CSV per downstream task
```

The raw `.dem` demos, trajectories, and event JSONs used to *generate* these files live alongside them; the full preparation pipeline is documented in [`doc/data-preparation.md`](doc/data-preparation.md).

---

## Repository tour

| Path | What's there |
|------|--------------|
| `main.py` | Single entry point — parses args, builds config, dispatches to a stage. |
| `configs/` | `train/` and `dev/` YAML configs for `contrastive` and `downstream`. |
| `src/models/` | `ContrastiveModel`, `LinearProbeModel`, and the `VideoEncoder` zoo (`modules/`). |
| `src/data_module/`, `src/dataset/` | Lightning DataModules and Datasets; video loading & masking. |
| `src/train/` | Training / test pipelines, trainer & callback setup. |
| `src/utils/` | Config, environment, experiment-dir, metric, and plotting helpers. |
| `src/scripts/` | Data preparation, SLURM job generation, results analysis, and visualization tooling. |
| `*.py` (root) | Experiment drivers and smoke tests (e.g. `train_all_downstream.py`, `test_all_tasks.py`). |

A `cheatsheet.md` in the repo root has copy-paste command snippets. `motivation.md` and `offset_explain.md` explain the research framing and the video-sync offset, respectively.

---

## Documentation (`doc/`)

The `doc/` folder contains human-readable deep-dives into each part of the codebase:

- **[`doc/overview.md`](doc/overview.md)** — high-level architecture and how a request flows through `main.py`.
- **[`doc/configuration.md`](doc/configuration.md)** — config files, override syntax, env vars, experiment directories.
- **[`doc/architecture.md`](doc/architecture.md)** — the contrastive & downstream models and the encoder zoo.
- **[`doc/data-pipeline.md`](doc/data-pipeline.md)** — DataModules, Datasets, video loading, UI/random masking, batching.
- **[`doc/training.md`](doc/training.md)** — training/test pipelines, Lightning trainer, checkpointing, resume.
- **[`doc/tasks.md`](doc/tasks.md)** — the downstream task registry and ML forms.
- **[`doc/data-preparation.md`](doc/data-preparation.md)** — turning raw demos/videos into labels & contrastive pairs.
- **[`doc/scripts.md`](doc/scripts.md)** — reference for the auxiliary scripts and experiment drivers.

---

## 🧩 Citation
If you use this repo, please cite our paper:
```bibtex
@article{wang2025x,
  title={X-Ego: Acquiring Team-Level Tactical Situational Awareness via Cross-Egocentric Contrastive Video Representation Learning},
  author={Wang, Yunzhe and Hans, Soham and Ustun, Volkan},
  journal={arXiv preprint arXiv:2510.19150},
  year={2025}
}
```
