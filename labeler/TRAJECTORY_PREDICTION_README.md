# Trajectory Prediction Labels

## Overview

The trajectory prediction task (`traj-gen`) predicts the future movements of all 10 players (5 teammates + 5 opponents) based on a 5-second video observation window.

## Data Structure

### Input
- **Video segment**: 5 seconds of gameplay from one player's POV

### Output
- **Trajectory**: 15 seconds total (5s observation + 10s future prediction)
- **Sampling rate**: 4 Hz (every 0.25 seconds)
- **Total timepoints**: 60 (15 seconds × 4 Hz)
- **Coordinates**: X, Y only (normalized)
- **Players**: All 10 players (sorted by steamid for consistency)

## File Format

### CSV File (`*_traj_prediction_*.csv`)

Contains metadata and indices to retrieve trajectories from H5 file:

**Key columns:**
- `idx`: Unique segment identifier
- `partition`: Data split (train/val/test)
- `match_id`, `round_num`: Game identifiers
- `map_name`: Map name
- `start_tick`, `video_end_tick`, `trajectory_end_tick`: Timing information
- `start_seconds`, `video_end_seconds`, `trajectory_end_seconds`: Time in seconds
- `h5_traj_idx`: Index to retrieve trajectory from H5 file
- `player_{i}_id`, `player_{i}_name`, `player_{i}_side`: Player metadata (i=0..9)

### H5 File (`*_traj_prediction_*.h5`)

Contains the actual trajectory data with compression:

**Dataset**: `trajectories`
- **Shape**: `(num_segments, 10, 60, 2)`
  - Dimension 0: Segment index
  - Dimension 1: Player index (sorted by steamid)
  - Dimension 2: Timepoint (0-59, representing 0-15 seconds at 4 Hz)
  - Dimension 3: Coordinates (0=X_norm, 1=Y_norm)

**Attributes**:
- `num_segments`: Total number of trajectory segments
- `num_players`: 10
- `num_timepoints`: 60
- `num_coords`: 2
- `coord_names`: ['X_norm', 'Y_norm']
- `trajectory_sample_rate`: 4 (Hz)
- `total_trajectory_sec`: 15 (seconds)
- `video_length_sec`: 5 (seconds)

## Usage Examples

### 1. Basic Loading

```python
import pandas as pd
import h5py

# Load metadata
df = pd.read_csv('labels/teammate_opponent_traj_prediction_5s_15s_4hz.csv')

# Load trajectories
with h5py.File('labels/teammate_opponent_traj_prediction_5s_15s_4hz.h5', 'r') as f:
    # Get trajectory for segment i
    segment_idx = df.iloc[0]['h5_traj_idx']
    trajectory = f['trajectories'][segment_idx]  # Shape: (10, 60, 2)
```

### 2. Preload All Data (Recommended)

For efficiency, preload the entire H5 file into memory:

```python
import numpy as np

# Preload all trajectories
with h5py.File('labels/teammate_opponent_traj_prediction_5s_15s_4hz.h5', 'r') as f:
    all_trajectories = f['trajectories'][:]  # Shape: (N, 10, 60, 2)

# Fast access by index
trajectory = all_trajectories[h5_traj_idx]
```

### 3. Batch Loading

```python
# Get batch of indices
batch_indices = df.iloc[0:32]['h5_traj_idx'].values

with h5py.File('labels/teammate_opponent_traj_prediction_5s_15s_4hz.h5', 'r') as f:
    batch_trajectories = f['trajectories'][batch_indices]  # Shape: (32, 10, 60, 2)
```

### 4. Access Specific Player Trajectory

```python
# Get trajectory for segment 0, player 3
trajectory = all_trajectories[0, 3, :, :]  # Shape: (60, 2)

# Split into observation and future
observation = trajectory[:20, :]  # First 5 seconds (5s × 4Hz = 20 points)
future = trajectory[20:, :]       # Next 10 seconds (10s × 4Hz = 40 points)

# Get X and Y coordinates
x_coords = trajectory[:, 0]  # Shape: (60,)
y_coords = trajectory[:, 1]  # Shape: (60,)
```

### 5. Separate by Team

```python
# Get player sides from CSV
row = df.iloc[0]
player_sides = [row[f'player_{i}_side'] for i in range(10)]

# Get trajectories
trajectory = all_trajectories[row['h5_traj_idx']]  # Shape: (10, 60, 2)

# Separate by team
ct_indices = [i for i, side in enumerate(player_sides) if side == 'ct']
t_indices = [i for i, side in enumerate(player_sides) if side == 't']

ct_trajectories = trajectory[ct_indices]  # Shape: (5, 60, 2)
t_trajectories = trajectory[t_indices]    # Shape: (5, 60, 2)
```

## Dataset Implementation

The project includes a complete dataset and data module implementation:

### Using the Dataset

```python
from omegaconf import OmegaConf
from dataset.teammate_opponent_traj_prediction import TeammateOpponentTrajPredictionDataset

# Load configuration
cfg = OmegaConf.load('configs/dev/teammate_opponent_traj_prediction.yaml')

# Create dataset
dataset = TeammateOpponentTrajPredictionDataset(cfg)

# Get a sample
sample = dataset[0]
# Returns:
# {
#   'video': Tensor[num_agents, num_frames, C, H, W],
#   'trajectories': Tensor[5, 60, 2],
#   'pov_team_side': str ('ct' or 't'),
#   'target_team_side': str ('ct' or 't'),
#   'agent_ids': List[str],
#   'target_player_ids': List[str]
# }
```

### Using the DataModule (Lightning)

```python
from data_module.teammate_opponent_traj_prediction import TeammateOpponentTrajPredictionDataModule
import lightning as L

# Load configuration
cfg = OmegaConf.load('configs/dev/teammate_opponent_traj_prediction.yaml')

# Create data module
data_module = TeammateOpponentTrajPredictionDataModule(cfg)

# Setup
data_module.prepare_data()
data_module.setup(stage='fit')

# Get dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Use with Lightning Trainer
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, data_module)
```

### POV-Target Team Sampling

The dataset uses **random sampling** for both POV and target team sides:

- POV side is randomly selected from {CT, T}
- Target side is independently randomly selected from {CT, T}
- This allows all 4 combinations with equal probability:
  - **CT→CT**: CT videos predicting CT trajectories (self-prediction)
  - **CT→T**: CT videos predicting T trajectories (opponent-prediction)
  - **T→CT**: T videos predicting CT trajectories (opponent-prediction)
  - **T→T**: T videos predicting T trajectories (self-prediction)

### Testing the Dataset

```bash
python scripts/data_analysis/test_traj_prediction_dataset.py
```

## Time Index Mapping

```python
# Convert timepoint index to seconds
def timepoint_to_seconds(timepoint_idx, sample_rate=4):
    return timepoint_idx / sample_rate

# Convert seconds to timepoint index
def seconds_to_timepoint(seconds, sample_rate=4):
    return int(seconds * sample_rate)

# Examples:
# timepoint 0  -> 0.00s (start)
# timepoint 20 -> 5.00s (end of video)
# timepoint 59 -> 14.75s (end of trajectory)
```

## Generation

To generate the labels:

```bash
# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run the labeler
python labeler/teammate_opponent_traj_prediction.py
```

Or customize parameters:

```python
from labeler.teammate_opponent_traj_prediction import TeammateOpponentTrajPredictionCreator

creator = TeammateOpponentTrajPredictionCreator(
    data_dir="path/to/data",
    output_dir="path/to/labels",
    partition_csv_path="path/to/partition.csv",
    stride_sec=1.0  # 1 second sliding window
)

creator.process_segments({
    'output_file_name': 'traj_pred.csv',
    'video_length_sec': 5,
    'total_trajectory_sec': 15,
    'trajectory_sample_rate': 4,
    'partition': ['train', 'val', 'test']
})
```

## Inspection

Use the inspection script to examine the generated data:

```bash
python scripts/inspect_labels/inspect_traj_prediction.py
```
