# Location Prediction Tasks

This module provides tools for creating labeled datasets for three location prediction tasks in Counter-Strike:

1. **Enemy Location Nowcast**: Predict enemy team locations at the current moment
2. **Enemy Location Forecast**: Predict enemy team locations in the future 
3. **Teammate Location Forecast**: Predict same team locations in the future

## Files

- `base.py`: Base class with common functionality for all location prediction tasks
- `enemy_location_nowcast.py`: Enemy location nowcast implementation
- `enemy_location_forecast.py`: Enemy location forecast implementation  
- `teammate_location_forecast.py`: Teammate location forecast implementation
- `test_location_prediction.py`: Test script demonstrating usage

## Quick Start

### 1. Enemy Location Nowcast

Predicts enemy team locations at the middle point of a video segment.

```python
from labeler.enemy_location_nowcast import EnemyLocationNowcastCreator

creator = EnemyLocationNowcastCreator(
    data_dir="data",
    output_dir="data/labels", 
    partition_csv_path="data/match_round_partitioned.csv"
)

creator.process_segments({
    'output_file_name': 'enemy_location_nowcast_5s.csv',
    'segment_length_sec': 5,
    'partition': ['train', 'val', 'test']
})
```

**Output CSV columns:**
- Basic info: idx, partition, match_id, round_num, map_name
- Timing: start_tick, end_tick, prediction_tick, normalized_*_seconds
- POV player: pov_player_id, pov_player_name, pov_player_side, pov_video_path
- Enemy locations: enemy_0_id, enemy_0_X, enemy_0_Y, enemy_0_Z, enemy_0_place, etc. (5 enemies)

### 2. Enemy Location Forecast

Predicts enemy team locations at a future moment (segment_duration + forecast_interval).

```python
from labeler.enemy_location_forecast import EnemyLocationForecastCreator

creator = EnemyLocationForecastCreator(
    data_dir="data",
    output_dir="data/labels",
    partition_csv_path="data/match_round_partitioned.csv"
)

creator.process_segments({
    'output_file_name': 'enemy_location_forecast_5s_10s.csv',
    'segment_length_sec': 5,
    'forecast_interval_sec': 10,  # Predict 10 seconds into future
    'partition': ['train', 'val', 'test']
})
```

**Output CSV columns:**
- Basic info: idx, partition, match_id, round_num, map_name
- Timing: start_tick, end_tick, forecast_tick, forecast_interval_sec, normalized_*_seconds
- POV player: pov_player_id, pov_player_name, pov_player_side, pov_video_path
- Enemy future locations: enemy_0_future_X, enemy_0_future_Y, enemy_0_future_Z, etc.

### 3. Teammate Location Forecast

Predicts same team locations at a future moment (excluding POV player).

```python
from labeler.teammate_location_forecast import TeammateLocationForecastCreator

creator = TeammateLocationForecastCreator(
    data_dir="data",
    output_dir="data/labels",
    partition_csv_path="data/match_round_partitioned.csv"
)

creator.process_segments({
    'output_file_name': 'teammate_location_forecast_5s_10s.csv', 
    'segment_length_sec': 5,
    'forecast_interval_sec': 10,  # Predict 10 seconds into future
    'partition': ['train', 'val', 'test']
})
```

**Output CSV columns:**
- Basic info: idx, partition, match_id, round_num, map_name
- Timing: start_tick, end_tick, forecast_tick, forecast_interval_sec, normalized_*_seconds
- POV player: pov_player_id, pov_player_name, pov_player_side, pov_video_path
- Teammate future locations: teammate_0_future_X, teammate_0_future_Y, etc. (4 teammates)

## Key Features

- **Sliding Window**: Generates segments with 1-tick step for maximum data coverage
- **Player Alive Filter**: Only includes segments where all players are alive throughout the observation/prediction window
- **Both Team Perspectives**: Creates segments from both T and CT player perspectives
- **Consistent Ordering**: Players sorted by steamid for reproducible results
- **Normalized Timing**: Includes normalized timestamps for video seeking
- **Partition Aware**: Follows the match_round_partitioned.csv split assignments

## Data Requirements

- Match-round partition CSV with columns: index, split, match_id, round_number
- Trajectory CSV files organized as: data/trajectory/{match_id}/{steamid}/round_{round_num}.csv
- Video files organized as: data/{video_folder}/{match_id}/{steamid}/round_{round_num}.mp4

## Configuration Options

All classes accept these parameters:
- `data_dir`: Path to data directory containing trajectory files
- `output_dir`: Directory for output CSV files  
- `partition_csv_path`: Path to match_round_partitioned.csv
- `trajectory_folder`: Trajectory subdirectory name (default: "trajectory")
- `video_folder`: Video subdirectory name (default: "video_544x306_30fps")
- `tick_rate`: Game tick rate (default: 64)
- `cpu_usage`: Multiprocessing CPU usage fraction (default: 0.9)

## Running Tests

```bash
python labeler/test_location_prediction.py
```

This will test all three location prediction tasks with sample configurations.
