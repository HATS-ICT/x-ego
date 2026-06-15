# Scripts Reference

Inventory of the auxiliary tooling: root-level experiment drivers and the `src/scripts/` subsystems. The data-preparation scripts get their own doc — see [data-preparation.md](data-preparation.md).

## Root-level drivers & smoke tests

All of these orchestrate `main.py`. **Stage 1 = contrastive**, **Stage 2 = downstream**.

| Script | Purpose |
|--------|---------|
| `train_all_downstream.py` | Production driver: runs Stage 2 on every benchmark task (`use_in_benchmark=yes` in `task_definitions.csv`) for a given map/model. Flags: `--map`, `--model-type`, `--ui-mask`, `--stage1-checkpoint`, `--extra-overrides`. |
| `run_clip_pipeline.sh` | Bash orchestrator: for each of mirage/dust2/inferno, runs CLIP Stage 1, finds the newest checkpoint, then runs downstream baseline + finetuned for seeds 1–2 via `train_all_downstream.py`. Robust `.env` loader (handles CRLF). |
| `run_performance_over_epoch.py` | Trains downstream tasks from Stage 1 checkpoints saved at different epochs to measure how pre-training duration affects downstream performance. Feeds the Pareto plots. |
| `run_missing_experiment.py` | Re-runs a curated `MISSING_EXPERIMENTS` list of `(model, task, init_type)` tuples to backfill gaps. |
| `evaluate_existing_output.py` | Walks `output/` and runs `--mode test` on every experiment lacking `test_results_*.json`. |
| `find_max_contra_accumulate_batches.py` | Memory-probing harness: finds the max physical `data.batch_size` and `contrastive_accumulate_batches` per encoder under a GPU-memory limit (uses dummy tensors + the real model; spawns subprocesses). |
| `check_model_param.py` | Diagnostic: instantiates each `VideoEncoder` and prints total/trainable/non-trainable param counts. |

### Smoke tests
Quick `dev`-mode checks that the pipeline runs end-to-end:

| Script | Checks |
|--------|--------|
| `test_all_model_setup.py` | Every model × map on `self_location_0s`, in three settings (contrastive-only, baseline downstream, downstream-from-checkpoint). |
| `test_all_tasks.py` | Baseline downstream (siglip2) across all benchmark tasks for dust2. |
| `test_all_model_setup_contrastive_only.py` | One contrastive accumulation window per model/map (canonical `(batch_size, accumulate)` table). |
| `test_all_tasks_after_contra.py` | Downstream runs loading real Stage 1 checkpoints (hardcoded `STAGE1_CHECKPOINTS`). |
| `test_dual_head_setup.py` | Contrastive for 4 encoders × {no mask, random mask}. |

## `src/scripts/` subsystems

### Data preparation & labels
- **`data_processing/`** — raw demos/videos → trajectories, events, offsets, partitions. See [data-preparation.md](data-preparation.md).
- **`task_creator/`** — label & contrastive-data generation. See [data-preparation.md](data-preparation.md) and [tasks.md](tasks.md).
- **`data_analysis/`** — dataset QA: `verify_data_integrity.py`, `get_data_stats.py`, `check_offset_coverage.py`, `get_offset_distribution.py`.
- **`inspect_labels/`** — `compute_class_weights.py` (class weights for location tasks), `plot_traj_heatmap.py`, label inspectors.
- **`scale_speed_repression/`** — speed-regression normalization (*"repression"* is a typo for *regression*): `compute_scalar.py` fits the scaler; `apply_normalization_to_existing_csv.py` applies it.
- **`control/`** — control-data trajectory parsing/inspection (`parse_traj_per_player.py`, `inspect_control_distribition.py`).

### Cluster experiment jobs
- **`create_exp_jobs/`** — SLURM batch-script generators. Current families: `main_contra_with_accu.py` (Stage 1 with accumulation), `downstream_finetuned.py`, `downstream_baseline.py`, `downstream_with_repeats.py`, plus `main_ui_cover.py`, `main_contra.py`, `main_dual_head.py`. `past_jobs/` holds superseded generators (historical).
- **`download_from_server/`** — scp-style pulls of checkpoints / results from the cluster to local. `download_checkpoint.py`, `download_downstream_result.py`.

### Results analysis & tables
- **`result_analysis/`** — `results_collector.py` (load/aggregate `test_results_*.json`), `table_printer.py`, `plotter.py` (+ plotting utils), `check_missing_experiment.py` (feeds `run_missing_experiment.py`), `check_metric_correctness.py`.
- **`print_table/`** — paper-table printers (`baseline.py`, `main_exp.py`, `main_exp_v2.py`).
- **`plot_pareto/`** — `plot_pareto.py` (pairs with `run_performance_over_epoch.py`).
- **`plot_label_distribution/`** — `plot_label_distribution_maps.py` (per-map label distributions).

### Visualization & interpretability
- **`attention_visualization/`** — render side-by-side attention overlay videos (`render_attention_video.py`, `render_team_attention_video.py`, batch drivers). Has its own README.
- **`contra_visualization/`** — t-SNE of the contrastive embedding space before/after training (`contra_tsne.py` and variants).
- **`contra_clustering/`** — lower-level embedding precompute to H5 + t-SNE (`embed_precompute.py`, `compute_single_tsne.py`).
- **`contra_cluster_v2/`** — newer single-file contrastive-space analysis (`analyze_contrastive_space.py`). Has a README.
- **`language_visualization/`** — SigLIP2 text↔image probing of egocentric vs allocentric understanding (`language_visualization.py`, `contrastive_space.py`, `concept_vocabulary.py`).
- **`parse_wiki/`** — build a CS2 wiki text corpus for the language probes (`run.py`, `fetch.py`, `parse.py`). Has a README.

## Notes

- Many scripts assume `.env` provides `DATA_BASE_PATH`, `OUTPUT_BASE_PATH`, `SRC_BASE_PATH` ([configuration.md](configuration.md#environment-variables)).
- Some carry Windows shims and hardcoded paths from a Windows-dev / Linux-cluster (SLURM) split; `past_jobs/` and `task_creator_helper/fix_*.py` are archives/one-offs, not maintained tooling.
