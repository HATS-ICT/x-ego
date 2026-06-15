# Overview

This document explains the high-level architecture of X-Ego and how a single command flows through the code. Read this first; the other docs in `doc/` zoom into individual subsystems.

## The two-stage idea

X-Ego learns **team-level situational awareness** from first-person (egocentric) gameplay video. The core hypothesis: if a video encoder is trained to map synchronized clips from *different teammates* (same game state, different viewpoints) to *similar* embeddings, those embeddings implicitly capture team coordination — and transfer to downstream tactical prediction tasks.

```
                      Stage 1                              Stage 2
              (cross-ego contrastive)               (downstream probing)

  team clips ──► VideoEncoder ──► projector ──► sigmoid    frozen VideoEncoder ──► linear head ──► task
  [agents,T,…]      │              contrastive loss             ▲ (loads Stage 1 weights)        prediction
                    └──────────── saved checkpoint ─────────────┘
```

- **Stage 1 (`--task contrastive`)** trains `ContrastiveModel` (`src/models/contrastive_model.py`). Positive pairs = agents from the same team/sample; loss = SigLIP-style sigmoid loss over an alignment matrix.
- **Stage 2 (`--task downstream`)** trains `LinearProbeModel` (`src/models/downstream.py`). The encoder is frozen (linear probe) — either loaded from a Stage 1 checkpoint (`model.stage1_checkpoint`) or used off-the-shelf (baseline) — and only a small head is trained on one task.

See [`architecture.md`](architecture.md) for model internals.

## The single entry point: `main.py`

Every run goes through `main.py`. Its job is to build a fully-resolved config and dispatch to one of four functions in `src/train/run_tasks.py`.

```
python main.py --mode {train,dev,test} --task {contrastive,downstream} [key.subkey=value ...]
```

Control flow (`main.py`):

1. **Parse args** — `--mode`, `--task`, optional `--config`, and any trailing `key=value` overrides.
2. **Build config** (branches on mode):
   - `test` mode → ignore config files; load `hparam.yaml` from the saved experiment named by `meta.resume_exp`, then apply CLI overrides.
   - `train`/`dev` mode → load `configs/train/<task>.yaml`; in `dev` mode merge `configs/dev/<task>.yaml`; apply CLI overrides. For `downstream`, `apply_task_config` fills in task metadata; for `contrastive`, `validate_contrastive_cfg` checks the config.
3. **Resolve paths** — `setup_base_pathing` injects `path.src/data/output` from env vars.
4. **Set up the experiment directory** — `setup_directory` either creates a fresh timestamped dir (and saves `hparam.yaml`) or resolves an existing one for resume/test.
5. **Dispatch** to `train_contrastive` / `test_contrastive` / `train_downstream` / `test_downstream`.

Config details are in [`configuration.md`](configuration.md); the dispatch targets and pipeline in [`training.md`](training.md).

## Code map

| Layer | Location | Doc |
|-------|----------|-----|
| Entry point / dispatch | `main.py`, `src/train/run_tasks.py` | [training.md](training.md) |
| Config & experiment setup | `src/utils/{config,env,experiment}_utils.py`, `configs/` | [configuration.md](configuration.md) |
| Models & encoders | `src/models/` (`contrastive_model.py`, `downstream.py`, `modules/`) | [architecture.md](architecture.md) |
| Data | `src/data_module/`, `src/dataset/` | [data-pipeline.md](data-pipeline.md) |
| Training/test pipeline | `src/train/` | [training.md](training.md) |
| Tasks | `src/scripts/task_creator/`, `apply_task_config` | [tasks.md](tasks.md) |
| Data preparation | `src/scripts/data_processing/`, `src/scripts/task_creator/` | [data-preparation.md](data-preparation.md) |
| Drivers, analysis, viz | root `*.py`, `src/scripts/*` | [scripts.md](scripts.md) |

## Mental model for a typical workflow

1. **Prepare data** once: raw `.dem` demos + recordings → trajectories, events, offsets, partitions → label CSVs and the contrastive CSV ([data-preparation.md](data-preparation.md)).
2. **Stage 1**: `python main.py --mode train --task contrastive` → produces a checkpoint under `OUTPUT_BASE_PATH/contrastive-<...>/checkpoint/`.
3. **Stage 2**: `python main.py --mode train --task downstream task.task_id=<task> model.stage1_checkpoint=<ckpt>` for each task — or use a driver like `train_all_downstream.py` to sweep all tasks.
4. **Analyze**: collect `test_results_*.json` across experiments ([scripts.md](scripts.md)).
