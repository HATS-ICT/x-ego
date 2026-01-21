"""
Create CSV for contrastive learning (Stage 1: Team Alignment).

This script samples video segments at a fixed stride (default 5s) from all
match-rounds in the train/val/test partition. No labels are created - only
metadata and teammate information needed for loading videos.

Usage:
    python -m scripts.task_creator.create_contrastive_data [--stride STRIDE] [--output_dir OUTPUT_DIR]
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))


class ContrastiveDataCreator:
    """
    Creates data CSV for contrastive learning.
    
    Unlike task-specific creators, this simply samples video segments
    at a fixed stride without creating any labels. The CSV contains
    only metadata and teammate information for loading videos.
    """
    
    def __init__(self, data_dir: str, output_dir: str, partition_csv_path: str,
                 trajectory_folder: str = "trajectory",
                 video_folder: str = "video_544x306_30fps",
                 tick_rate: int = 64, seed: int = 42,
                 stride_sec: float = 5.0, segment_length_sec: float = 5.0):
        """
        Initialize ContrastiveDataCreator.
        
        Args:
            data_dir: Path to the data directory
            output_dir: Directory where output CSV files will be saved
            partition_csv_path: Path to the match round partition CSV file
            trajectory_folder: Name of trajectory folder
            video_folder: Name of video folder
            tick_rate: Game tick rate (default: 64)
            seed: Random seed for reproducibility
            stride_sec: Step size in seconds between segments (default: 5.0)
            segment_length_sec: Length of each segment in seconds (default: 5.0)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.partition_csv_path = partition_csv_path
        self.trajectory_folder = trajectory_folder
        self.video_folder = video_folder
        self.tick_rate = tick_rate
        self.seed = seed
        self.stride_sec = stride_sec
        self.segment_length_sec = segment_length_sec
        
        np.random.seed(seed)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_partition_data()
        
        # Cache for loaded data
        self._trajectory_cache = {}
    
    def _load_partition_data(self):
        """Load match round partition data from CSV."""
        partition_path = Path(self.partition_csv_path)
        
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition CSV not found: {partition_path}")
        
        self.partition_df = pd.read_csv(partition_path)
        print(f"Loaded {len(self.partition_df)} match-round entries from partition file")
    
    def _load_player_trajectories(self, match_id: str, round_num: int) -> Dict[str, pd.DataFrame]:
        """Load trajectory data for all players in a specific match and round."""
        cache_key = (match_id, round_num)
        if cache_key in self._trajectory_cache:
            return self._trajectory_cache[cache_key]
        
        match_dir = self.data_dir / self.trajectory_folder / match_id
        player_trajectories = {}
        
        if not match_dir.exists():
            return player_trajectories
        
        player_dirs = [d for d in match_dir.iterdir() if d.is_dir()]
        
        for player_dir in player_dirs:
            steamid = player_dir.name
            round_file = player_dir / f"round_{round_num}.csv"
            
            if round_file.exists():
                try:
                    df = pd.read_csv(round_file, keep_default_na=False)
                    if not df.empty:
                        player_trajectories[steamid] = df
                except Exception as e:
                    print(f"Error loading trajectory for player {steamid}, round {round_num}: {e}")
        
        self._trajectory_cache[cache_key] = player_trajectories
        return player_trajectories
    
    def _extract_player_data_at_tick(self, df: pd.DataFrame, target_tick: int) -> Dict:
        """Extract player data at a specific tick."""
        exact_data = df[df['tick'] == target_tick]
        if not exact_data.empty:
            row = exact_data.iloc[0]
            return {
                'steamid': row.get('steamid'),
                'name': row.get('name'),
                'side': row.get('side'),
                'X': row.get('X'),
                'Y': row.get('Y'),
                'Z': row.get('Z'),
                'X_norm': row.get('X_norm'),
                'Y_norm': row.get('Y_norm'),
                'Z_norm': row.get('Z_norm'),
                'place': row.get('place'),
                'health': row.get('health')
            }
        
        # Try to find closest tick within a small window
        window_size = 5
        window_data = df[(df['tick'] >= target_tick - window_size) &
                         (df['tick'] <= target_tick + window_size)]
        
        if not window_data.empty:
            closest_idx = (window_data['tick'] - target_tick).abs().idxmin()
            row = df.loc[closest_idx]
            return {
                'steamid': row.get('steamid'),
                'name': row.get('name'),
                'side': row.get('side'),
                'X': row.get('X'),
                'Y': row.get('Y'),
                'Z': row.get('Z'),
                'X_norm': row.get('X_norm'),
                'Y_norm': row.get('Y_norm'),
                'Z_norm': row.get('Z_norm'),
                'place': row.get('place'),
                'health': row.get('health')
            }
        
        return None
    
    def _find_player_death_tick(self, player_df: pd.DataFrame) -> int:
        """Find the tick when a specific player dies, or max tick if they survive."""
        if player_df.empty or 'health' not in player_df.columns:
            return player_df['tick'].max() if not player_df.empty else 0
        
        death_mask = player_df['health'] <= 0
        if death_mask.any():
            first_death_idx = death_mask.idxmax()
            return player_df.loc[first_death_idx, 'tick']
        return player_df['tick'].max()
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                      partition: str) -> List[Dict]:
        """
        Extract segments from a specific round.
        
        Samples video segments at fixed stride for all team combinations.
        """
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 2:
            return []
        
        segments = []
        segment_ticks = int(self.segment_length_sec * self.tick_rate)
        stride_ticks = int(self.stride_sec * self.tick_rate)
        stride_ticks = max(1, stride_ticks)
        
        # Get global tick range and round start tick
        all_min_ticks = []
        all_max_ticks = []
        for df in player_trajectories.values():
            if not df.empty:
                all_min_ticks.append(df['tick'].min())
                all_max_ticks.append(df['tick'].max())
        
        if not all_min_ticks:
            return []
        
        global_min_tick = min(all_min_ticks)
        global_max_tick = max(all_max_ticks)
        
        # Get map name
        map_name = 'de_mirage'
        for df in player_trajectories.values():
            if 'map_name' in df.columns and not df.empty:
                map_name = df.iloc[0]['map_name']
                break
        
        # Group players by side
        t_players = {}
        ct_players = {}
        
        for steamid, df in player_trajectories.items():
            if df.empty or 'side' not in df.columns:
                continue
            side = df.iloc[0]['side']
            if side == 't':
                t_players[steamid] = df
            elif side == 'ct':
                ct_players[steamid] = df
        
        # Process each side as a team
        for pov_side, team_trajectories in [('T', t_players), ('CT', ct_players)]:
            if len(team_trajectories) < 2:
                continue
            
            # Sample segments at fixed stride
            current_tick = global_min_tick
            
            while current_tick + segment_ticks <= global_max_tick:
                end_tick = current_tick + segment_ticks
                middle_tick = current_tick + segment_ticks // 2
                
                # Get alive teammates at this tick
                teammates = []
                for steamid, df in team_trajectories.items():
                    player_data = self._extract_player_data_at_tick(df, middle_tick)
                    if player_data and player_data.get('health', 0) > 0:
                        teammates.append(player_data)
                
                # Need at least 2 alive teammates for contrastive learning
                if len(teammates) >= 2:
                    # Sort teammates by steamid for consistency
                    teammates.sort(key=lambda x: x['steamid'])
                    
                    # Compute time in seconds
                    start_seconds = current_tick / self.tick_rate
                    end_seconds = end_tick / self.tick_rate
                    prediction_seconds = middle_tick / self.tick_rate
                    
                    # Normalized times (relative to round start)
                    norm_start_seconds = (current_tick - global_min_tick) / self.tick_rate
                    norm_end_seconds = (end_tick - global_min_tick) / self.tick_rate
                    norm_pred_seconds = (middle_tick - global_min_tick) / self.tick_rate
                    
                    segment_info = {
                        'partition': partition,
                        'pov_team_side': pov_side.lower(),
                        'seg_duration_sec': self.segment_length_sec,
                        'start_tick': current_tick - global_min_tick,
                        'end_tick': end_tick - global_min_tick,
                        'prediction_tick': middle_tick - global_min_tick,
                        'start_seconds': start_seconds,
                        'end_seconds': end_seconds,
                        'prediction_seconds': prediction_seconds,
                        'normalized_start_seconds': norm_start_seconds,
                        'normalized_end_seconds': norm_end_seconds,
                        'normalized_prediction_seconds': norm_pred_seconds,
                        'match_id': match_id,
                        'round_num': round_num,
                        'map_name': map_name,
                        'teammates_data': teammates,
                        'num_alive_teammates': len(teammates)
                    }
                    segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict]) -> pd.DataFrame:
        """Create output CSV from segments."""
        output_rows = []
        idx = 0
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'pov_team_side': segment['pov_team_side'],
                'seg_duration_sec': segment['seg_duration_sec'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'prediction_tick': segment['prediction_tick'],
                'start_seconds': segment['start_seconds'],
                'end_seconds': segment['end_seconds'],
                'prediction_seconds': segment['prediction_seconds'],
                'normalized_start_seconds': segment['normalized_start_seconds'],
                'normalized_end_seconds': segment['normalized_end_seconds'],
                'normalized_prediction_seconds': segment['normalized_prediction_seconds'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name'],
                'num_alive_teammates': segment['num_alive_teammates']
            }
            
            # Add teammate data (up to 5 teammates)
            for i in range(5):
                if i < len(segment['teammates_data']):
                    tm = segment['teammates_data'][i]
                    row[f'teammate_{i}_id'] = tm['steamid']
                    row[f'teammate_{i}_name'] = tm['name']
                    row[f'teammate_{i}_side'] = tm['side']
                    row[f'teammate_{i}_X'] = tm['X_norm']
                    row[f'teammate_{i}_Y'] = tm['Y_norm']
                    row[f'teammate_{i}_Z'] = tm['Z_norm']
                    row[f'teammate_{i}_place'] = tm['place']
                    row[f'teammate_{i}_health'] = tm['health']
                else:
                    # Mark as dead/missing
                    row[f'teammate_{i}_id'] = ''
                    row[f'teammate_{i}_name'] = ''
                    row[f'teammate_{i}_side'] = ''
                    row[f'teammate_{i}_X'] = ''
                    row[f'teammate_{i}_Y'] = ''
                    row[f'teammate_{i}_Z'] = ''
                    row[f'teammate_{i}_place'] = ''
                    row[f'teammate_{i}_health'] = 0
            
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num', 'start_tick'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df
    
    def process(self, partitions: List[str] = None, output_filename: str = "contrastive.csv") -> pd.DataFrame:
        """
        Process all match-rounds and create contrastive data CSV.
        
        Args:
            partitions: List of partitions to include ['train', 'val', 'test']
            output_filename: Name of the output CSV file
            
        Returns:
            DataFrame with processed data
        """
        if partitions is None:
            partitions = ['train', 'val', 'test']
        
        print("=" * 60)
        print("Contrastive Data Creation")
        print("=" * 60)
        print(f"Stride: {self.stride_sec}s")
        print(f"Segment length: {self.segment_length_sec}s")
        print(f"Partitions: {partitions}")
        
        filtered_partition_df = self.partition_df[
            self.partition_df['split'].isin(partitions)
        ]
        
        print(f"Found {len(filtered_partition_df)} match-round combinations")
        
        if filtered_partition_df.empty:
            print("No match-round combinations found. Exiting.")
            return pd.DataFrame()
        
        print("\nExtracting segments...")
        
        all_segments = []
        
        for _, row in tqdm(filtered_partition_df.iterrows(),
                          total=len(filtered_partition_df),
                          desc="Processing match-rounds"):
            match_id = row['match_id']
            round_num = row['round_number']
            partition = row['split']
            
            try:
                segments = self._extract_segments_from_round(match_id, round_num, partition)
                all_segments.extend(segments)
            except Exception as e:
                print(f"Error processing match {match_id}, round {round_num}: {e}")
                continue
        
        print(f"\nExtracted {len(all_segments)} total segments")
        
        if not all_segments:
            print("No valid segments found. Exiting.")
            return pd.DataFrame()
        
        print("\nCreating output CSV...")
        df = self._create_output_csv(all_segments)
        
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)
        
        print(f"\nSaved {len(df)} segments to {output_path}")
        
        if len(df) > 0:
            print("\nSummary by partition:")
            for partition in df['partition'].unique():
                partition_data = df[df['partition'] == partition]
                print(f"  {partition}: {len(partition_data)} segments")
            
            print("\nSummary by num_alive_teammates:")
            for n in sorted(df['num_alive_teammates'].unique()):
                count = len(df[df['num_alive_teammates'] == n])
                print(f"  {n} teammates: {count} segments ({count/len(df)*100:.1f}%)")
        
        return df
    
    def clear_cache(self):
        """Clear trajectory cache."""
        self._trajectory_cache.clear()


def main():
    parser = argparse.ArgumentParser(description="Create contrastive learning data CSV")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: DATA_BASE_PATH/labels)")
    parser.add_argument("--output_filename", type=str, default="contrastive.csv",
                        help="Output filename (default: contrastive.csv)")
    parser.add_argument("--stride", type=float, default=5.0,
                        help="Stride between segments in seconds (default: 5.0)")
    parser.add_argument("--segment_length", type=float, default=5.0,
                        help="Segment length in seconds (default: 5.0)")
    parser.add_argument("--partitions", type=str, nargs="+", default=None,
                        help="Partitions to include (default: train val test)")
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    DATA_BASE_PATH = os.getenv('DATA_BASE_PATH')
    if not DATA_BASE_PATH:
        print("ERROR: DATA_BASE_PATH environment variable not set")
        sys.exit(1)
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(DATA_BASE_PATH, 'labels')
    
    partition_csv = os.path.join(DATA_BASE_PATH, 'match_round_partitioned.csv')
    
    creator = ContrastiveDataCreator(
        data_dir=DATA_BASE_PATH,
        output_dir=output_dir,
        partition_csv_path=partition_csv,
        stride_sec=args.stride,
        segment_length_sec=args.segment_length
    )
    
    partitions = args.partitions if args.partitions else ['train', 'val', 'test']
    
    creator.process(partitions=partitions, output_filename=args.output_filename)


if __name__ == "__main__":
    main()
