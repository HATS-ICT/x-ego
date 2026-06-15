# Data Preparation

How raw Counter-Strike 2 demos and gameplay recordings become the trajectories, offsets, partitions, and label CSVs that training consumes. This is offline tooling — run once per dataset, not per experiment.

Relevant code: `src/scripts/data_processing/`, `src/scripts/task_creator/`.

## Pipeline overview

```
.dem demos + POV recordings
        │
        ▼  src/scripts/data_processing/
  parse events / trajectories ─► downsample & resize video ─► compute sync offsets ─► partition ─► fit scalers
        │
        ▼  src/scripts/task_creator/
  create_all_labels.py  ──►  labels/all_tasks/<task_id>.csv   (+ task_definitions.csv)
  create_contrastive_data.py  ──►  labels/contrastive.csv
        │
        ▼
   main.py  (Stage 1 / Stage 2)
```

## Step 1 — `src/scripts/data_processing/`

Approximate run order:

| Script | Role |
|--------|------|
| `parse_event.py` | Parse `.dem` files (via `awpy.Demo`) into per-round event JSON. |
| `parse_traj_per_player.py` | Parse demos into per-player trajectory CSVs (positions/ticks). |
| `save_4fps_video.py` | Downsample 30 fps recordings to 4 fps (ffmpeg), preserving folder structure. |
| `resize_distort.py` | Resize clips to a square aspect (e.g. 306×306). |
| `compute_offset.py` | Compute the video↔game-time `offset_sec` per round → `time_offset.json`. See `offset_explain.md`. |
| `create_data_partition.py` | Assign match-rounds to train/val/test (70/15/15). |
| `fit_trajectory_scalar.py` | Fit a `MinMaxScaler` over trajectories; append normalized coordinate columns. |
| `create_location_prediction_labels.py` | Generate enemy/teammate location nowcast/forecast labels across partitions. |
| `embed_video.py` | (optional) Pre-compute video embeddings per encoder for caching. |

Inspection helpers: `check_unique_locations_from_traj.py` (list place names per map).

> Several scripts have Windows-specific shims and the occasional hardcoded path — they reflect a Windows-dev / Linux-cluster split. `remove_half_labels.py` is a one-off downsampling utility, not part of the standard flow.

## Step 2 — `src/scripts/task_creator/`

This is the label-generation hub. See [tasks.md](tasks.md) for the resulting task catalog.

| Script | Role |
|--------|------|
| `create_all_labels.py` | **Master generator.** Instantiates every creator in `TASK_CONFIGS` and writes one `<task_id>.csv` per task per map. Run as `python -m src.scripts.task_creator.create_all_labels`. |
| `create_contrastive_data.py` | Builds the Stage 1 `contrastive.csv` by sampling video segments at a fixed stride across match-rounds, recording teammate grouping (no labels). |
| `task_definitions.py` | Task enums (`TaskCategory`, `MLForm`, `TemporalType`, `DataSource`), the `TaskDefinition` dataclass, and the map place/direction/weapon vocabularies. |
| `analyze_label_stats.py`, `plot_label_distribution.py` | Per-task label statistics and distribution plots. |
| `task_creator_helper/` | Implementation library (not entry points): `base_task_creator.py` plus per-category creators — `location_tasks*.py`, `combat_tasks.py`, `coordination_tasks*.py`, `bomb_tasks*.py`, `spatial_tasks.py`. |

The `task_creator_helper/fix_*.py` scripts are historical one-off source migrations (they regex-rewrite the other helpers) — not maintained tooling.

## Resulting on-disk layout

```
<DATA_BASE_PATH>/<map>/
├── time_offset.json
├── video_306x306_4fps/<match_id>/<player_id>/round_<nn>.mp4
└── labels/
    ├── contrastive.csv
    ├── task_definitions.csv
    └── all_tasks/<task_id>.csv
```

This is exactly what the runtime loaders in [data-pipeline.md](data-pipeline.md) expect.

## Related analysis tooling

`src/scripts/data_analysis/` validates and characterizes the prepared dataset — e.g. `verify_data_integrity.py` (train/val/test references resolve to real files), `get_data_stats.py` (totals such as video duration), and `check_offset_coverage.py` / `get_offset_distribution.py` (audit `time_offset.json`). See [scripts.md](scripts.md) for the broader script inventory.
