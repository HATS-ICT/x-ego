import sys
import os
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from labeler.base import LocationPredictionBase

load_dotenv()


class TeammateOpponentTrajPredictionCreator(LocationPredictionBase):
    """
    Creates labeled segments for trajectory prediction task.
    
    Takes a 5 sec video segment and predicts trajectories for all 10 players
    over a 15 second window (5 sec observation + 10 sec future prediction).
    
    Trajectories are sampled at 4 Hz (60 time points total) and saved to an H5 file
    with only X, Y coordinates. The CSV contains metadata and indices to retrieve
    trajectories from the H5 file.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store trajectory data during extraction
        self.trajectory_data = []
    
    def _extract_segments_from_round(self, match_id: str, round_num: int, 
                                   cfg: Dict[str, Any]) -> List[Dict]:
        """
        Extract trajectory segments from a specific round.
        
        Args:
            match_id: Match identifier
            round_num: Round number
            cfg: Configuration dictionary containing all parameters
        
        Returns:
            List of segment dictionaries with trajectory data
        """
        video_length_sec = cfg['video_length_sec']  # 5 seconds
        total_trajectory_sec = cfg['total_trajectory_sec']  # 15 seconds
        trajectory_sample_rate = cfg['trajectory_sample_rate']  # 4 Hz
        
        # Load all player trajectories for this round
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) != 10:
            return []  # Must have exactly 10 players
        
        segments = []
        total_trajectory_ticks = total_trajectory_sec * self.tick_rate
        
        # Find valid tick range where all players are alive
        min_tick, max_tick_alive = self._get_valid_tick_range(player_trajectories)
        
        if max_tick_alive - min_tick < total_trajectory_ticks:
            return []  # Not enough ticks for trajectory
        
        # Generate segments with configurable stride
        stride_ticks = int(self.stride_sec * self.tick_rate)
        stride_ticks = max(1, stride_ticks)
        current_tick = min_tick
        
        # Calculate sampling interval in ticks (4 Hz = every 0.25 sec = every 16 ticks at 64 Hz)
        sample_interval_ticks = int(self.tick_rate / trajectory_sample_rate)
        num_timepoints = int(total_trajectory_sec * trajectory_sample_rate)
        
        while current_tick + total_trajectory_ticks <= max_tick_alive:
            video_end_tick = current_tick + video_length_sec * self.tick_rate
            trajectory_end_tick = current_tick + total_trajectory_ticks
            
            # Check if all players are alive throughout the entire trajectory window
            all_players_alive = True
            for df in player_trajectories.values():
                window_data = df[(df['tick'] >= current_tick) & (df['tick'] <= trajectory_end_tick)]
                if window_data.empty or (window_data['health'] <= 0).any():
                    all_players_alive = False
                    break
            
            if not all_players_alive:
                current_tick += stride_ticks
                continue
            
            # Extract trajectory data for all 10 players at 4 Hz
            # Shape: (10 players, 60 timepoints, 2 coords)
            trajectory_array = np.zeros((10, num_timepoints, 2), dtype=np.float32)
            all_players_metadata = []
            
            # Get all players sorted by steamid for consistent ordering
            sorted_players = sorted(player_trajectories.items(), key=lambda x: x[0])
            
            valid_trajectory = True
            for player_idx, (steamid, df) in enumerate(sorted_players):
                # Extract metadata at start tick
                player_metadata = self._extract_player_data_at_tick(df, current_tick)
                if not player_metadata:
                    valid_trajectory = False
                    break
                all_players_metadata.append(player_metadata)
                
                # Extract trajectory at 4 Hz
                for time_idx in range(num_timepoints):
                    sample_tick = current_tick + time_idx * sample_interval_ticks
                    player_data = self._extract_player_data_at_tick(df, sample_tick)
                    
                    if player_data:
                        trajectory_array[player_idx, time_idx, 0] = player_data['X_norm']
                        trajectory_array[player_idx, time_idx, 1] = player_data['Y_norm']
                    else:
                        valid_trajectory = False
                        break
                
                if not valid_trajectory:
                    break
            
            if valid_trajectory and len(all_players_metadata) == 10:
                # Get map name
                map_name = 'unknown'
                for df in player_trajectories.values():
                    if 'map_name' in df.columns and not df.empty:
                        map_name = df.iloc[0]['map_name']
                        break
                
                # Store trajectory data
                h5_traj_idx = len(self.trajectory_data)
                self.trajectory_data.append(trajectory_array)
                
                segment_info = {
                    'start_tick': current_tick,
                    'video_end_tick': video_end_tick,
                    'trajectory_end_tick': trajectory_end_tick,
                    'start_seconds': current_tick / self.tick_rate,
                    'video_end_seconds': video_end_tick / self.tick_rate,
                    'trajectory_end_seconds': trajectory_end_tick / self.tick_rate,
                    'normalized_start_seconds': (current_tick - min_tick) / self.tick_rate,
                    'normalized_video_end_seconds': (video_end_tick - min_tick) / self.tick_rate,
                    'normalized_trajectory_end_seconds': (trajectory_end_tick - min_tick) / self.tick_rate,
                    'video_length_sec': video_length_sec,
                    'total_trajectory_sec': total_trajectory_sec,
                    'trajectory_sample_rate': trajectory_sample_rate,
                    'num_timepoints': num_timepoints,
                    'map_name': map_name,
                    'all_players_metadata': all_players_metadata,
                    'h5_traj_idx': h5_traj_idx
                }
                segments.append(segment_info)
            
            # Move to next step based on stride
            current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], cfg: Dict[str, Any]) -> pd.DataFrame:
        """
        Create the final CSV output and save trajectories to H5 file.
        
        The CSV contains metadata and h5_traj_idx to retrieve trajectories.
        The H5 file contains trajectory data with shape (num_segments, 10, 60, 2).
        """
        output_rows = []
        idx = 0
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'video_length_sec': segment['video_length_sec'],
                'total_trajectory_sec': segment['total_trajectory_sec'],
                'trajectory_sample_rate': segment['trajectory_sample_rate'],
                'num_timepoints': segment['num_timepoints'],
                'start_tick': segment['start_tick'],
                'video_end_tick': segment['video_end_tick'],
                'trajectory_end_tick': segment['trajectory_end_tick'],
                'start_seconds': segment['start_seconds'],
                'video_end_seconds': segment['video_end_seconds'],
                'trajectory_end_seconds': segment['trajectory_end_seconds'],
                'normalized_start_seconds': segment['normalized_start_seconds'],
                'normalized_video_end_seconds': segment['normalized_video_end_seconds'],
                'normalized_trajectory_end_seconds': segment['normalized_trajectory_end_seconds'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name'],
                'h5_traj_idx': segment['h5_traj_idx']
            }
            
            # Add all players' metadata (10 players total)
            for i, player in enumerate(segment['all_players_metadata']):
                row[f'player_{i}_id'] = player['steamid']
                row[f'player_{i}_name'] = player['name']
                row[f'player_{i}_side'] = player['side']
            
            output_rows.append(row)
            idx += 1
        
        # Create DataFrame and sort
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'], 
                               ascending=[True, True, True])
            df = df.reset_index(drop=True)
            # Update idx after sorting
            df['idx'] = range(len(df))
        
        # Save trajectory data to H5 file
        if self.trajectory_data:
            h5_filename = cfg['output_file_name'].replace('.csv', '.h5')
            h5_path = self.output_dir / h5_filename
            
            print(f"\nSaving trajectory data to {h5_path}")
            print(f"  Total segments: {len(self.trajectory_data)}")
            print(f"  Shape per segment: (10 players, {cfg['total_trajectory_sec'] * cfg['trajectory_sample_rate']} timepoints, 2 coords)")
            
            # Stack all trajectories into a single array
            # Shape: (num_segments, 10, 60, 2)
            trajectory_array = np.stack(self.trajectory_data, axis=0)
            
            with h5py.File(h5_path, 'w') as f:
                # Create dataset with compression
                f.create_dataset(
                    'trajectories',
                    data=trajectory_array,
                    compression='gzip',
                    compression_opts=4
                )
                
                # Store metadata as attributes
                f['trajectories'].attrs['num_segments'] = len(self.trajectory_data)
                f['trajectories'].attrs['num_players'] = 10
                f['trajectories'].attrs['num_timepoints'] = trajectory_array.shape[2]
                f['trajectories'].attrs['num_coords'] = 2
                f['trajectories'].attrs['coord_names'] = ['X_norm', 'Y_norm']
                f['trajectories'].attrs['trajectory_sample_rate'] = cfg['trajectory_sample_rate']
                f['trajectories'].attrs['total_trajectory_sec'] = cfg['total_trajectory_sec']
                f['trajectories'].attrs['video_length_sec'] = cfg['video_length_sec']
            
            print(f"  H5 file saved successfully")
            print(f"  File size: {h5_path.stat().st_size / (1024*1024):.2f} MB")
        
        return df
    
    def process_segments(self, cfg: Dict[str, Any]):
        """
        Main method to process segments and create labeled data.
        
        Args:
            cfg: Configuration dictionary with keys:
                - output_file_name: Name of the output CSV file (H5 will use same name)
                - video_length_sec: Length of video segment (default: 5)
                - total_trajectory_sec: Total trajectory duration (default: 15)
                - trajectory_sample_rate: Sampling rate in Hz (default: 4)
                - partition: List of partitions to include ['train', 'val', 'test']
        """
        # Reset trajectory data for fresh processing
        self.trajectory_data = []
        
        # Set defaults
        cfg.setdefault('video_length_sec', 5)
        cfg.setdefault('total_trajectory_sec', 15)
        cfg.setdefault('trajectory_sample_rate', 4)
        
        # Add segment_length_sec for base class compatibility
        # (it's the same as video_length_sec for this task)
        cfg['segment_length_sec'] = cfg['video_length_sec']
        
        # Call parent's process_segments
        return super().process_segments(cfg)


def main():
    """Main function for testing trajectory prediction label creation."""
    # Load paths from environment variables
    DATA_BASE_PATH = os.getenv('DATA_BASE_PATH')
    if not DATA_BASE_PATH:
        raise ValueError("DATA_BASE_PATH environment variable not set. Please check your .env file.")
    
    DATA_DIR = DATA_BASE_PATH
    OUTPUT_DIR = os.path.join(DATA_BASE_PATH, 'labels')
    PARTITION_CSV_PATH = os.path.join(DATA_BASE_PATH, 'match_round_partitioned.csv')
    
    # Create trajectory prediction creator
    creator = TeammateOpponentTrajPredictionCreator(
        DATA_DIR,
        OUTPUT_DIR, 
        PARTITION_CSV_PATH,
        cpu_usage=0.9,
        stride_sec=1.0  # 1 second stride by default
    )
    
    # Process segments
    creator.process_segments({
        'output_file_name': 'teammate_opponent_traj_prediction_5s_15s_4hz.csv',
        'video_length_sec': 5,          # 5 seconds of video
        'total_trajectory_sec': 15,      # 15 seconds total (5 sec observation + 10 sec future)
        'trajectory_sample_rate': 4,     # 4 Hz sampling = 60 timepoints
        'partition': ['train', 'val', 'test']
    })


if __name__ == "__main__":
    main()
