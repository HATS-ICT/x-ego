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
from typing import Dict, List, Tuple
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count

sys.path.append(str(Path(__file__).parent.parent.parent))


def _load_player_trajectories(data_dir: Path, trajectory_folder: str,
                               match_id: str, round_num: int) -> Dict[str, pd.DataFrame]:
    """Load trajectory data for all players in a specific match and round."""
    match_dir = data_dir / trajectory_folder / match_id
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
            except Exception:
                pass
    
    return player_trajectories


def _is_player_alive_in_segment(df: pd.DataFrame, start_tick: int, end_tick: int) -> bool:
    """
    Check if a player is alive throughout the entire segment.
    
    A player is considered dead if their health drops to 0 at any point
    within the segment [start_tick, end_tick].
    """
    segment_data = df[(df['tick'] >= start_tick) & (df['tick'] <= end_tick)]
    
    if segment_data.empty:
        return False
    
    if 'health' not in segment_data.columns:
        return True
    
    # Player is alive only if health > 0 throughout the entire segment
    return (segment_data['health'] > 0).all()


def _get_player_steamid(df: pd.DataFrame) -> str:
    """Get the steamid from a player's trajectory dataframe."""
    if df.empty or 'steamid' not in df.columns:
        return None
    return df.iloc[0]['steamid']


def _process_single_round(args: Tuple) -> List[Dict]:
    """
    Process a single match-round. This is the worker function for multiprocessing.
    
    Args:
        args: Tuple of (match_id, round_num, partition, data_dir, trajectory_folder,
                       tick_rate, stride_sec, segment_length_sec)
    
    Returns:
        List of segment dictionaries
    """
    (match_id, round_num, partition, data_dir, trajectory_folder,
     tick_rate, stride_sec, segment_length_sec) = args
    
    data_dir = Path(data_dir)
    player_trajectories = _load_player_trajectories(data_dir, trajectory_folder,
                                                     match_id, round_num)
    
    if len(player_trajectories) < 2:
        return []
    
    segments = []
    segment_ticks = int(segment_length_sec * tick_rate)
    stride_ticks = int(stride_sec * tick_rate)
    stride_ticks = max(1, stride_ticks)
    
    # Get global tick range
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
            
            # Get teammates alive throughout the entire segment
            teammates = []
            for steamid, df in team_trajectories.items():
                if _is_player_alive_in_segment(df, current_tick, end_tick):
                    teammates.append({'steamid': steamid})
            
            # Need at least 2 alive teammates for contrastive learning
            if len(teammates) >= 2:
                # Sort teammates by steamid for consistency
                teammates.sort(key=lambda x: x['steamid'])
                
                # Normalized times (relative to round start)
                norm_start_seconds = (current_tick - global_min_tick) / tick_rate
                norm_end_seconds = (end_tick - global_min_tick) / tick_rate
                
                segment_info = {
                    'partition': partition,
                    'pov_team_side': pov_side.lower(),
                    'seg_duration_sec': segment_length_sec,
                    'start_tick': current_tick - global_min_tick,
                    'end_tick': end_tick - global_min_tick,
                    'start_seconds': norm_start_seconds,
                    'end_seconds': norm_end_seconds,
                    'match_id': match_id,
                    'round_num': round_num,
                    'map_name': map_name,
                    'teammates_data': teammates,
                    'num_alive_teammates': len(teammates)
                }
                segments.append(segment_info)
            
            current_tick += stride_ticks
    
    return segments


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
                 stride_sec: float = 5.0, segment_length_sec: float = 5.0,
                 num_workers: int = None):
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
            num_workers: Number of worker processes (default: cpu_count)
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
        self.num_workers = num_workers if num_workers else cpu_count()
        
        np.random.seed(seed)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_partition_data()
    
    def _load_partition_data(self):
        """Load match round partition data from CSV."""
        partition_path = Path(self.partition_csv_path)
        
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition CSV not found: {partition_path}")
        
        self.partition_df = pd.read_csv(partition_path)
        print(f"Loaded {len(self.partition_df)} match-round entries from partition file")
    
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
                'start_seconds': segment['start_seconds'],
                'end_seconds': segment['end_seconds'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name'],
                'num_alive_teammates': segment['num_alive_teammates']
            }
            
            # Add teammate steamids (up to 5 teammates)
            for i in range(5):
                if i < len(segment['teammates_data']):
                    tm = segment['teammates_data'][i]
                    row[f'teammate_{i}_id'] = tm['steamid']
                else:
                    row[f'teammate_{i}_id'] = ''
            
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
        print(f"Workers: {self.num_workers}")
        
        filtered_partition_df = self.partition_df[
            self.partition_df['split'].isin(partitions)
        ]
        
        print(f"Found {len(filtered_partition_df)} match-round combinations")
        
        if filtered_partition_df.empty:
            print("No match-round combinations found. Exiting.")
            return pd.DataFrame()
        
        # Prepare arguments for multiprocessing
        work_items = []
        for _, row in filtered_partition_df.iterrows():
            work_items.append((
                row['match_id'],
                row['round_number'],
                row['split'],
                str(self.data_dir),
                self.trajectory_folder,
                self.tick_rate,
                self.stride_sec,
                self.segment_length_sec
            ))
        
        print("\nExtracting segments with multiprocessing...")
        
        all_segments = []
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_single_round, work_items),
                total=len(work_items),
                desc="Processing match-rounds"
            ))
        
        for segments in results:
            all_segments.extend(segments)
        
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


def main():
    parser = argparse.ArgumentParser(description="Create contrastive learning data CSV")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: DATA_BASE_PATH/labels)")
    parser.add_argument("--output_filename", type=str, default="contrastive.csv",
                        help="Output filename (default: contrastive.csv)")
    parser.add_argument("--stride", type=float, default=3.0,
                        help="Stride between segments in seconds (default: 5.0)")
    parser.add_argument("--segment_length", type=float, default=5.0,
                        help="Segment length in seconds (default: 5.0)")
    parser.add_argument("--partitions", type=str, nargs="+", default=None,
                        help="Partitions to include (default: train val test)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count)")
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
        segment_length_sec=args.segment_length,
        num_workers=args.num_workers
    )
    
    partitions = args.partitions if args.partitions else ['train', 'val', 'test']
    
    creator.process(partitions=partitions, output_filename=args.output_filename)


if __name__ == "__main__":
    main()
