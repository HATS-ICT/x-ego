import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
import multiprocessing as mp
from tqdm import tqdm
from abc import ABC, abstractmethod

class LocationPredictionBase(ABC):
    """
    Base class for location prediction tasks including:
    - Enemy location nowcast (current moment)
    - Enemy location forecast (future moment) 
    - Teammate location forecast (future moment)
    
    Handles common functionality like trajectory loading, segment extraction,
    and player death boundary detection.
    """
    
    def __init__(self, data_dir: str, output_dir: str, partition_csv_path: str,
                 trajectory_folder: str = "trajectory", video_folder: str = "video_544x306_30fps", 
                 tick_rate: int = 64, seed: int = 42, cpu_usage: float = 0.9, stride_sec: float = 1.0):
        """
        Initialize the LocationPredictionBase.
        
        Args:
            data_dir: Path to the data directory containing trajectory and video files
            output_dir: Directory where output CSV files will be saved
            partition_csv_path: Path to the match round partition CSV file
            trajectory_folder: Name of trajectory folder (default: "trajectory")
            video_folder: Name of video folder (default: "video_544x306_30fps")
            tick_rate: Game tick rate (default: 64)
            seed: Random seed for reproducibility (default: 42)
            cpu_usage: Fraction of CPU cores to use for multiprocessing (default: 0.9)
            stride_sec: Step size in seconds between segments (default: 1.0)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.partition_csv_path = partition_csv_path
        self.trajectory_folder = trajectory_folder
        self.video_folder = video_folder
        self.tick_rate = tick_rate
        self.seed = seed
        self.cpu_usage = cpu_usage
        
        # Validate and set stride
        if stride_sec <= 0.0:
            raise ValueError("stride_sec must be > 0.0")
        self.stride_sec = stride_sec
        
        # Calculate number of processes to use
        self.num_processes = max(1, int(mp.cpu_count() * cpu_usage))
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load partition data
        self._load_partition_data()
        
        # Debug flag
        self.debug = False
    
    def _load_partition_data(self):
        """Load match round partition data from CSV."""
        partition_path = Path(self.partition_csv_path)
        
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition CSV not found: {partition_path}")
        
        self.partition_df = pd.read_csv(partition_path)
        print(f"Loaded {len(self.partition_df)} match-round entries from partition file")
    
    def _load_player_trajectories(self, match_id: str, round_num: int) -> Dict[str, pd.DataFrame]:
        """
        Load trajectory data for all players in a specific match and round.
        
        Args:
            match_id: Match identifier
            round_num: Round number
        
        Returns:
            Dict mapping player steamid to their trajectory DataFrame
        """
        match_dir = self.data_dir / self.trajectory_folder / match_id
        player_trajectories = {}
        
        if not match_dir.exists():
            if self.debug:
                print(f"Match directory does not exist: {match_dir}")
            return player_trajectories
        
        # Get all player directories
        player_dirs = [d for d in match_dir.iterdir() if d.is_dir()]
        
        if self.debug:
            print(f"Found {len(player_dirs)} player directories for match {match_id}, round {round_num}")
        
        for player_dir in player_dirs:
            steamid = player_dir.name
            round_file = player_dir / f"round_{round_num}.csv"
            
            if round_file.exists():
                try:
                    df = pd.read_csv(round_file, keep_default_na=False)
                    if not df.empty:
                        player_trajectories[steamid] = df
                        if self.debug:
                            print(f"Loaded trajectory for player {steamid}: {len(df)} rows")
                except Exception as e:
                    print(f"Error loading trajectory for player {steamid}, round {round_num}: {e}")
            elif self.debug:
                print(f"Round file does not exist: {round_file}")
        
        if self.debug:
            print(f"Total loaded trajectories: {len(player_trajectories)}")
        
        return player_trajectories
    
    def _find_first_death_tick(self, player_trajectories: Dict[str, pd.DataFrame]) -> int:
        """
        Find the tick when the first player dies across all players.
        
        Args:
            player_trajectories: Dict mapping steamid to trajectory DataFrame
        
        Returns:
            Tick when first player dies, or max_tick if no deaths
        """
        if not player_trajectories:
            return 0
        
        death_ticks = []
        
        for df in player_trajectories.values():
            if not df.empty and 'health' in df.columns:
                # Find first tick where health drops to 0 or below
                death_mask = df['health'] <= 0
                if death_mask.any():
                    first_death_idx = death_mask.idxmax()
                    death_tick = df.loc[first_death_idx, 'tick']
                    death_ticks.append(death_tick)
        
        if death_ticks:
            return min(death_ticks)  # Return tick of first death
        else:
            # No deaths found, return maximum tick available
            all_ticks = []
            for df in player_trajectories.values():
                if not df.empty:
                    all_ticks.extend(df['tick'].tolist())
            return max(all_ticks) if all_ticks else 0
    
    def _get_players_by_side(self, player_trajectories: Dict[str, pd.DataFrame], 
                           target_side: str) -> Dict[str, pd.DataFrame]:
        """
        Filter players by team side (t or ct).
        
        Args:
            player_trajectories: Dict mapping steamid to trajectory DataFrame
            target_side: Target side ('t' or 'ct')
        
        Returns:
            Dict mapping steamid to trajectory DataFrame for players on target_side
        """
        side_players = {}
        
        for steamid, df in player_trajectories.items():
            if not df.empty and 'side' in df.columns:
                player_side = df.iloc[0]['side']
                if player_side == target_side:
                    side_players[steamid] = df
        
        return side_players
    
    def _get_opposite_side(self, side: str) -> str:
        """Get the opposite team side."""
        return 'ct' if side == 't' else 't'
    
    def _get_valid_tick_range(self, player_trajectories: Dict[str, pd.DataFrame]) -> Tuple[int, int]:
        """
        Get the valid tick range where all players have data and are alive.
        
        Args:
            player_trajectories: Dict mapping steamid to trajectory DataFrame
        
        Returns:
            Tuple of (min_tick, max_tick_before_death)
        """
        if not player_trajectories:
            return 0, 0
        
        # Find common tick range across all players
        all_ticks = []
        for df in player_trajectories.values():
            if not df.empty:
                all_ticks.extend(df['tick'].tolist())
        
        if not all_ticks:
            return 0, 0
        
        min_tick = min(all_ticks)
        max_tick = max(all_ticks)
        
        # Find first death to limit the range
        first_death_tick = self._find_first_death_tick(player_trajectories)
        max_tick_alive = min(max_tick, first_death_tick)
        
        return min_tick, max_tick_alive
    
    def _extract_player_data_at_tick(self, df: pd.DataFrame, target_tick: int) -> Optional[Dict]:
        """
        Extract player data at a specific tick, with fallback to nearest tick.
        
        Args:
            df: Player trajectory DataFrame
            target_tick: Target tick
        
        Returns:
            Dict with player data or None if not found
        """
        # Try exact tick first
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
                'place': row.get('place'),
                'health': row.get('health')
            }
        # Try to find closest tick within a small window
        window_size = 5
        window_data = df[(df['tick'] >= target_tick - window_size) & 
                        (df['tick'] <= target_tick + window_size)]
        
        if not window_data.empty:
            # Get closest tick
            closest_idx = (window_data['tick'] - target_tick).abs().idxmin()
            row = df.loc[closest_idx]
            return {
                'steamid': row.get('steamid'),
                'name': row.get('name'),
                'side': row.get('side'),
                'X': row.get('X'),
                'Y': row.get('Y'), 
                'Z': row.get('Z'),
                'place': row.get('place'),
                'health': row.get('health')
            }
        
        return None
    
    def _construct_video_path(self, match_id: str, steamid: str, round_num: int) -> str:
        """Construct video path for a player's round using pathlib for cross-platform compatibility."""
        return str((Path("data") / self.video_folder / match_id / steamid / f"round_{round_num}.mp4"))
    
    @abstractmethod
    def _extract_segments_from_round(self, match_id: str, round_num: int, 
                                   cfg: Dict[str, Any]) -> List[Dict]:
        """
        Extract segments from a specific round. Must be implemented by subclasses.
        
        Args:
            match_id: Match identifier
            round_num: Round number
            cfg: Configuration dictionary containing all parameters
        
        Returns:
            List of segment dictionaries
        """
        pass
    
    @abstractmethod
    def _create_output_csv(self, all_segments: List[Dict], cfg: Dict[str, Any]) -> pd.DataFrame:
        """
        Create the final CSV output. Must be implemented by subclasses.
        
        Args:
            all_segments: List of all extracted segments
            cfg: Configuration dictionary
        
        Returns:
            DataFrame with the created CSV data
        """
        pass
    
    def process_segments(self, cfg: Dict[str, Any]):
        """
        Main method to process segments and create labeled data.
        
        Args:
            cfg: Configuration dictionary with keys:
                - output_file_name: Name of the output CSV file
                - segment_length_sec: Length of segments in seconds
                - partition: List of partitions to include ['train', 'val', 'test']
                - Additional parameters specific to each task
        """
        # Validate configuration
        required_keys = ['output_file_name', 'segment_length_sec', 'partition']
        for key in required_keys:
            if key not in cfg:
                raise ValueError(f"Missing required configuration key: {key}")
        
        print("Processing segments with configuration:")
        print(f"  Segment length: {cfg.segment_length_sec} seconds")
        print(f"  Partitions: {cfg.partition}")
        
        # Filter partition data for desired partitions
        filtered_partition_df = self.partition_df[
            self.partition_df['split'].isin(cfg.partition)
        ]
        
        print(f"Found {len(filtered_partition_df)} match-round combinations in desired partitions")
        
        if filtered_partition_df.empty:
            print("No match-round combinations found for specified partitions. Exiting.")
            return
        
        # Extract segments using multiprocessing
        print(f"\nExtracting segments using {self.num_processes} processes...")
        print(f"  CPU usage: {self.cpu_usage*100:.0f}% ({self.num_processes}/{mp.cpu_count()} cores)")
        
        all_segments = []
        
        # Process each match-round combination
        for _, row in tqdm(filtered_partition_df.iterrows(), 
                          total=len(filtered_partition_df), 
                          desc="Processing match-rounds"):
            match_id = row['match_id']
            round_num = row['round_number']
            partition = row['split']
            
            try:
                segments = self._extract_segments_from_round(match_id, round_num, cfg)
                
                # Add partition info to each segment
                for segment in segments:
                    segment['partition'] = partition
                    segment['match_id'] = match_id
                    segment['round_num'] = round_num
                
                all_segments.extend(segments)
                
            except Exception as e:
                print(f"Error processing match {match_id}, round {round_num}: {e}")
                continue
        
        print(f"Extracted {len(all_segments)} total segments")
        
        if not all_segments:
            print("No valid segments found. Exiting.")
            return
        
        # Create output CSV
        print("\nCreating output CSV...")
        df = self._create_output_csv(all_segments, cfg)
        
        output_path = self.output_dir / cfg.output_file_name
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(df)} segments to {output_path}")
        
        # Print summary statistics
        if len(df) > 0:
            print("\nSummary by partition:")
            for partition in df['partition'].unique():
                partition_data = df[df['partition'] == partition]
                print(f"  {partition}: {len(partition_data)} segments")
            
            print(f"\nTotal segments: {len(df)}")
            print(f"Matches processed: {df['match_id'].nunique()}")
            print(f"Rounds covered: {df['round_num'].nunique()}")
        
        return df
