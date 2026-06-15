# Data Pipeline

How clips and labels become batched tensors. This is the *runtime* loading path; for how the underlying CSVs/videos are produced, see [data-preparation.md](data-preparation.md).

Relevant code: `src/data_module/`, `src/dataset/`.

## DataModules

`BaseDataModule` (`src/data_module/base.py`, a `LightningDataModule`) holds the shared logic:

- builds the label path `<path.data>/<map>/<labels_folder>/<labels_filename>` and validates it,
- splits partitions with Polars (`df.filter(pl.col('partition') == split)`),
- builds dataloaders from `cfg.data` (`batch_size`, `num_workers`, `pin_mem`, `persistent_workers`, `prefetch_factor`),
- supports resumable sampler state (`state_dict`/`load_state_dict`).

Two concrete subclasses:

| DataModule | Dataset | Collate | Used by |
|------------|---------|---------|---------|
| `ContrastiveDataModule` | `ContrastiveDataset` | `contrastive_collate_fn` | Stage 1 |
| `DownstreamDataModule` | `DownstreamDataset` | `downstream_collate_fn` | Stage 2 |

## Contrastive data (Stage 1)

`ContrastiveDataset` (`src/dataset/contrastive.py`) reads `contrastive.csv` (metadata + teammate IDs `teammate_0_id..teammate_4_id` + `num_alive_teammates`; **no labels**). The first `num_alive_teammates` slots are the alive agents.

For each sample it loads one clip per alive agent and returns:

```python
{'videos': [<[T,C,H,W]> per agent], 'num_agents': int,
 'pov_team_side': str, 'pov_team_side_encoded': tensor,
 'agent_ids': [...], 'original_csv_idx': ...}
```

### `FixedVideoBatchSampler`

The key trick for Stage 1: `data.batch_size` is a budget of **total videos**, not samples. `FixedVideoBatchSampler` (`src/data_module/contrastive.py`) packs whole teams into that budget (truncating the last team if it overflows) and yields `(dataset_index, agents_to_load)` pairs. This keeps GPU memory bounded regardless of how many teammates are alive in a given round. Shuffling is per-epoch seeded; val/test use `drop_last=True`.

### `contrastive_collate_fn`

Flattens all agents across the batch into a single tensor and records the per-sample counts (matching `ContrastiveModel`'s no-padding design — see [architecture.md](architecture.md#variable-agent-batching-no-padding)):

```python
{'video': [total_agents, T, C, H, W],
 'agent_counts': [B],
 'agent_ids': [...], 'pov_team_side': [...],
 'pov_team_side_encoded': tensor, 'original_csv_idx': [...]}
```

## Downstream data (Stage 2)

`DownstreamDataset` (`src/dataset/downstream.py`) is one agent per sample. It reads `<task_id>.csv` from `labels/all_tasks`, filters by partition, and produces a label tensor per the task's `ml_form`:

```python
{'video': [T,C,H,W], 'label': ...,
 'pov_team_side': str, 'pov_team_side_encoded': tensor,
 'original_csv_idx': ..., 'match_id': str, 'player_id': str}
```

- `cfg.task.label_column` may be a single column (`label`) or a `;`-separated list (`label_0;label_1;…`) for multi-output tasks. `_validate_label_values` range-checks `multi_cls` labels and column counts for `multi_label_cls`.
- Tick→second conversion uses `tick_rate = 64`; the POV player comes from `pov_steamid`; team side is read from the `pov_side` column (note: contrastive uses `pov_team_side`).

`downstream_collate_fn` default-collates tensors and keeps `match_id`/`player_id`/`pov_team_side` as string lists.

## Shared video loading (`src/dataset/dataset_utils.py`)

Both datasets use the same loading utilities:

- **`construct_video_path`** → `<data>/<map>/<video_folder>/<match_id>/<player_id>/round_<round_num>.mp4`.
- **`load_video_clip`** uses [decord](https://github.com/dmlc/decord)'s `VideoReader`. It samples `T = fixed_duration_seconds × target_fps` frames evenly (`np.linspace`), applies the per-round time offset, optionally adds `time_jitter_max_seconds`, and returns `[T,C,H,W]` half-precision tensors.
- **Masking** is applied during loading:
  - **UI mask** (`data.ui_mask`): `minimap_only`, `all` (HUD + minimap), or `none`.
  - **Random tube mask** (`data.random_mask`, Stage 1 default-on): masks `num_tubes` rectangular regions, consistent across frames, sized to cover ~30–70% of area — an augmentation.
- **`transform_video`** normalizes per the encoder's processor. `init_video_processor` returns the right processor per `model_type`:
  - `siglip2/clip/dinov2/dinov3` → `AutoImageProcessor` (`image`)
  - `vivit` → `VivitImageProcessor` (`video_frames`)
  - `videomae` → `VideoMAEImageProcessor` (`video_frames`)
  - `vjepa2` → `VJEPA2VideoProcessor` (`video`)
  - `resnet50` → ImageNet mean/std at 224 (`torchvision_image`)

## Time offsets

Each map has a `time_offset.json` (`<data>/<map>/time_offset.json`) giving the per match/agent/round `offset_sec` that aligns the recorded video to game time. `ContrastiveDataset` loads it eagerly and **fails fast if it's missing**. The meaning of `offset_sec` is explained in `offset_explain.md` at the repo root; it is computed by `src/scripts/data_processing/compute_offset.py` ([data-preparation.md](data-preparation.md)).
