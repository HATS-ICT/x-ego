import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import random
import multiprocessing as mp
from tqdm import tqdm
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ctfm.labeler.label_creator import SegmentLabelCreator

DATA_PARTITION = "full"


class MultiAgentFutureLocationLabelCreator(SegmentLabelCreator):
    """
    A specialized label creator for multi-agent future location prediction.
    
    Extracts 5-second segments for a single team (5 players) and predicts their
    location K seconds into the future. Focuses on periods where all team members
    are alive and filters for de_mirage map only.
    """
    
    def __init__(self, data_dir: str, output_dir: str, partition_csv_path: str,
                 trajectory_folder: str = "trajectory", video_folder: str = "video_544x306", 
                 tick_rate: int = 64, seed: int = 42, example_video_save_path: str = None,
                 cpu_usage: float = 0.9, future_prediction_sec: int = 10):
        """Initialize MultiAgentFutureLocationLabelCreator with future prediction parameter."""
        super().__init__(data_dir, output_dir, partition_csv_path, trajectory_folder, 
                        video_folder, tick_rate, seed, example_video_save_path, cpu_usage)
        self.future_prediction_sec = future_prediction_sec
    
    def _is_mirage_match(self, match_id: str) -> bool:
        """
        Check if a match is on de_mirage by examining the first player's first round.
        
        Args:
            match_id: Match identifier (e.g., "1-1f5f9f17-0111-4e9c-bb82-e39d3e9020e6-1-1")
        
        Returns:
            True if the match is on de_mirage, False otherwise
        """
        match_dir = Path(self.data_dir) / "recording" / "trajectory" / match_id
        if not match_dir.exists():
            return False
        
        # Get first player directory
        player_dirs = [d for d in match_dir.iterdir() if d.is_dir()]
        if not player_dirs:
            return False
        
        first_player_dir = player_dirs[0]
        round_1_file = first_player_dir / "round_1.csv"
        
        if not round_1_file.exists():
            return False
        
        try:
            # Read just the first row to check map name
            df = pd.read_csv(round_1_file, nrows=1, keep_default_na=False)
            if df.empty or 'map_name' not in df.columns:
                return False
            
            map_name = df.iloc[0]['map_name']
            return map_name == 'de_mirage'
        except Exception as e:
            print(f"Error checking map for match {match_id}: {e}")
            return False
    
    def _get_all_match_ids(self) -> List[str]:
        """Get all match IDs from the trajectory directory."""
        trajectory_dir = Path(self.data_dir) / "recording" / "trajectory"
        if not trajectory_dir.exists():
            return []
        
        match_dirs = [d.name for d in trajectory_dir.iterdir() if d.is_dir()]
        return match_dirs
    
    def _load_team_trajectories(self, match_id: str, round_num: int, team_side: str) -> Dict[str, pd.DataFrame]:
        """
        Load trajectory data for all players in a specific team, match and round.
        
        Args:
            match_id: Match identifier
            round_num: Round number
            team_side: Team side ('t' or 'ct')
        
        Returns:
            Dict mapping player steamid to their trajectory DataFrame
        """
        match_dir = Path(self.data_dir) / "recording" / "trajectory" / match_id
        team_trajectories = {}
        
        if not match_dir.exists():
            return team_trajectories
        
        # Get all player directories
        player_dirs = [d for d in match_dir.iterdir() if d.is_dir()]
        
        for player_dir in player_dirs:
            steamid = player_dir.name
            round_file = player_dir / f"round_{round_num}.csv"
            
            if round_file.exists():
                try:
                    df = pd.read_csv(round_file, keep_default_na=False)
                    if not df.empty and 'side' in df.columns:
                        # Check if this player belongs to the specified team
                        player_side = df.iloc[0]['side']
                        if player_side == team_side:
                            team_trajectories[steamid] = df
                except Exception as e:
                    print(f"Error loading trajectory for player {steamid}, round {round_num}: {e}")
        
        return team_trajectories
    
    def _find_team_death_boundary(self, team_trajectories: Dict[str, pd.DataFrame]) -> int:
        """
        Find the tick when the first team member dies.
        
        Args:
            team_trajectories: Dict mapping steamid to trajectory DataFrame
        
        Returns:
            Tick when first team member dies, or max_tick if no deaths
        """
        if not team_trajectories:
            return 0
        
        death_ticks = []
        
        for df in team_trajectories.values():
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
            for df in team_trajectories.values():
                if not df.empty:
                    all_ticks.extend(df['tick'].tolist())
            return max(all_ticks) if all_ticks else 0
    
    def _extract_team_future_segments(self, match_id: str, round_num: int, team_side: str,
                                    segment_length_sec: int, future_prediction_sec: int) -> List[Dict]:
        """
        Extract segments for a single team with future location predictions.
        
        Args:
            match_id: Match identifier
            round_num: Round number
            team_side: Team side ('t' or 'ct')
            segment_length_sec: Segment duration in seconds
            future_prediction_sec: Future prediction time in seconds
        
        Returns:
            List of segment dictionaries with future location data
        """
        team_trajectories = self._load_team_trajectories(match_id, round_num, team_side)
        
        if len(team_trajectories) != 5:
            return []  # Must have exactly 5 players in team
        
        # Find when first team member dies
        death_boundary_tick = self._find_team_death_boundary(team_trajectories)
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        future_ticks = future_prediction_sec * self.tick_rate
        total_window_ticks = segment_ticks + future_ticks
        
        # Find common tick range across all team players
        all_ticks = []
        for df in team_trajectories.values():
            if not df.empty:
                all_ticks.extend(df['tick'].tolist())
        
        if not all_ticks:
            return []
        
        min_tick = min(all_ticks)
        max_tick = min(max(all_ticks), death_boundary_tick)
        
        # Generate rolling window segments
        current_tick = min_tick
        while current_tick + total_window_ticks <= max_tick:
            segment_end_tick = current_tick + segment_ticks
            future_tick = current_tick + total_window_ticks
            
            # Check if all 5 team players have data for the entire window and are alive
            segment_player_data = {}
            future_player_data = {}
            all_players_valid = True
            
            for steamid, df in team_trajectories.items():
                # Check segment data (5 seconds)
                segment_data = df[(df['tick'] >= current_tick) & (df['tick'] <= segment_end_tick)]
                if segment_data.empty or (segment_data['health'] <= 0).any():
                    all_players_valid = False
                    break
                
                # Check future data point (at future_tick)
                future_data = df[df['tick'] == future_tick]
                if future_data.empty:
                    # Try to find closest tick to future_tick
                    future_window = df[(df['tick'] >= future_tick - 5) & (df['tick'] <= future_tick + 5)]
                    if future_window.empty:
                        all_players_valid = False
                        break
                    future_data = future_window.iloc[[0]]  # Take first available
                
                # Get segment representative data (middle of segment)
                mid_idx = len(segment_data) // 2
                mid_data = segment_data.iloc[mid_idx]
                
                segment_player_data[steamid] = {
                    'steamid': steamid,
                    'name': mid_data['name'],
                    'side': mid_data['side']
                }
                
                # Get future location data
                future_row = future_data.iloc[0]
                future_player_data[steamid] = {
                    'future_X': future_row['X'],
                    'future_Y': future_row['Y'],
                    'future_Z': future_row['Z'],
                    'future_place': future_row['place']
                }
            
            if all_players_valid and len(segment_player_data) == 5:
                # Sort players by steamid for consistent ordering
                sorted_steamids = sorted(segment_player_data.keys())
                
                # Create segment info
                first_player_df = list(team_trajectories.values())[0]
                min_tick_in_trajectory = first_player_df['tick'].min()
                
                segment_info = {
                    'start_tick': current_tick,
                    'end_tick': segment_end_tick,
                    'future_tick': future_tick,
                    'start_seconds': current_tick / self.tick_rate,
                    'end_seconds': segment_end_tick / self.tick_rate,
                    'normalized_start_seconds': (current_tick - min_tick_in_trajectory) / self.tick_rate,
                    'normalized_end_seconds': (segment_end_tick - min_tick_in_trajectory) / self.tick_rate,
                    'normalized_future_seconds': (future_tick - min_tick_in_trajectory) / self.tick_rate,
                    'duration_seconds': segment_length_sec,
                    'future_prediction_seconds': future_prediction_sec,
                    'map_name': mid_data['map_name'],
                    'round_num': mid_data['round_num'],
                    'team_side': team_side,
                    'match_id': match_id,
                    'players': [segment_player_data[steamid] for steamid in sorted_steamids],
                    'future_locations': [future_player_data[steamid] for steamid in sorted_steamids]
                }
                
                segments.append(segment_info)
            
            # Move to next tick (no overlap for now, can be added later)
            current_tick += 1
        
        return segments
    
    def _extract_match_future_segments(self, match_id: str, segment_length_sec: int,
                                     future_prediction_sec: int) -> List[Dict]:
        """
        Extract all future prediction segments from a match across all rounds and teams.
        
        Args:
            match_id: Match identifier
            segment_length_sec: Segment duration in seconds
            future_prediction_sec: Future prediction time in seconds
        
        Returns:
            List of segment dictionaries
        """
        if not self._is_mirage_match(match_id):
            return []  # Skip non-mirage matches
        
        all_segments = []
        
        # Try rounds 1-16 (typical CS:GO match length)
        for round_num in range(1, 17):
            # Process both teams
            for team_side in ['t', 'ct']:
                segments = self._extract_team_future_segments(
                    match_id, round_num, team_side, segment_length_sec, future_prediction_sec
                )
                all_segments.extend(segments)
        
        return all_segments
    
    @staticmethod
    def _process_match_worker(args):
        """Worker function for multiprocessing match extraction."""
        (match_id, data_dir, trajectory_folder, segment_length_sec, future_prediction_sec, tick_rate) = args
        
        try:
            # Create a temporary instance for processing
            temp_creator = MultiAgentFutureLocationLabelCreator.__new__(MultiAgentFutureLocationLabelCreator)
            temp_creator.data_dir = Path(data_dir)
            temp_creator.trajectory_folder = trajectory_folder
            temp_creator.tick_rate = tick_rate
            temp_creator.future_prediction_sec = future_prediction_sec
            
            # Bind methods
            temp_creator._is_mirage_match = MultiAgentFutureLocationLabelCreator._is_mirage_match.__get__(temp_creator)
            temp_creator._load_team_trajectories = MultiAgentFutureLocationLabelCreator._load_team_trajectories.__get__(temp_creator)
            temp_creator._find_team_death_boundary = MultiAgentFutureLocationLabelCreator._find_team_death_boundary.__get__(temp_creator)
            temp_creator._extract_team_future_segments = MultiAgentFutureLocationLabelCreator._extract_team_future_segments.__get__(temp_creator)
            
            return temp_creator._extract_match_future_segments(match_id, segment_length_sec, future_prediction_sec)
            
        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            return []
    
    def _create_future_location_csv(self, all_segments: List[Dict], config: Dict[str, Any]):
        """Create the final CSV output for future location prediction data."""
        output_rows = []
        idx = 0
        
        # Load partition data once
        partition_df = pd.read_csv(self.partition_csv_path)
        video_to_partition = dict(zip(partition_df['video_path'], partition_df['split']))
        
        for segment in all_segments:
            # Determine partition based on first player's video path
            first_player = segment['players'][0]
            first_steamid = first_player['steamid']
            match_id = segment['match_id']
            round_num = segment['round_num']
            
            # Construct video path for partition lookup
            if os.name == 'nt':  # Windows
                sample_video_path = f"data\\{DATA_PARTITION}\\recording\\{self.video_folder}\\{match_id}\\{first_steamid}\\round_{round_num}.mp4"
            else:  # Unix-like systems
                sample_video_path = f"data/{DATA_PARTITION}/recording/{self.video_folder}/{match_id}/{first_steamid}/round_{round_num}.mp4"
            partition = video_to_partition.get(sample_video_path, 'unknown')
            
            if partition not in config.get('partition', ['train', 'val', 'test']):
                continue  # Skip if not in desired partitions
            
            row = {
                'idx': idx,
                'partition': partition,
                'seg_duration_sec': segment['duration_seconds'],
                'future_interval_sec': segment['future_prediction_seconds'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'future_tick': segment['future_tick'],
                'start_seconds': segment['start_seconds'],
                'end_seconds': segment['end_seconds'],
                'future_seconds': segment['future_tick'] / self.tick_rate,
                'normalized_start_seconds': segment['normalized_start_seconds'],
                'normalized_end_seconds': segment['normalized_end_seconds'],
                'normalized_future_seconds': segment['normalized_future_seconds'],
                'match_id': match_id,
                'round_num': round_num,
                'map_name': segment['map_name'],
                'team_side': segment['team_side']
            }
            
            # Add data for all 5 team players
            for i, (player, future_location) in enumerate(zip(segment['players'], segment['future_locations'])):
                steamid = player['steamid']
                if os.name == 'nt':  # Windows
                    video_path = f"data\\{DATA_PARTITION}\\recording\\{self.video_folder}\\{match_id}\\{steamid}\\round_{round_num}.mp4"
                else:  # Unix-like systems
                    video_path = f"data/{DATA_PARTITION}/recording/{self.video_folder}/{match_id}/{steamid}/round_{round_num}.mp4"
                
                # Player info
                row[f'player_{i}_id'] = steamid
                row[f'player_{i}_video_path'] = video_path
                row[f'player_{i}_name'] = player['name']
                
                # Future location
                row[f'player_{i}_future_X'] = future_location['future_X']
                row[f'player_{i}_future_Y'] = future_location['future_Y']
                row[f'player_{i}_future_Z'] = future_location['future_Z']
                row[f'player_{i}_future_place'] = future_location['future_place']
            
            output_rows.append(row)
            idx += 1
        
        # Create DataFrame and sort
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'team_side', 'match_id', 'round_num'], ascending=[True, True, True, True])
            df = df.reset_index(drop=True)
            # Update idx after sorting
            df['idx'] = range(len(df))
        
        output_path = self.output_dir / config['output_file_name']
        df.to_csv(output_path, index=False)
        
        print(f"\nSaved {len(output_rows)} future location prediction segments to {output_path}")
        
        # Print summary statistics
        if len(df) > 0:
            print("\nSummary by partition:")
            for partition in df['partition'].unique():
                partition_data = df[df['partition'] == partition]
                print(f"  {partition}: {len(partition_data)} segments")
            
            print("\nSummary by team:")
            for team in df['team_side'].unique():
                team_data = df[df['team_side'] == team]
                print(f"  {team}: {len(team_data)} segments")
            
            print("\nOverall summary:")
            print(f"  Total segments: {len(df)}")
            print(f"  Matches processed: {df['match_id'].nunique()}")
            print(f"  Rounds covered: {df['round_num'].nunique()}")
            print(f"  Future prediction time: {self.future_prediction_sec} seconds")
        else:
            print("\nNo segments were saved - check partition matching and de_mirage filtering.")
        
        return df
    
    def parse_future_location_predictions(self, config: Dict[str, Any]):
        """
        Parse future location prediction data and create labeled segments.
        
        Args:
            config: Configuration dictionary with keys:
                - output_file_name: Name of the output CSV file
                - samples_per_type_train/val/test: Max samples for each partition
                - segment_length_sec: Length of segments in seconds
                - future_prediction_sec: Future prediction time in seconds (default: 10)
                - partition: List of partitions to include ['train', 'val', 'test']
        """
        # Validate configuration
        required_keys = ['output_file_name', 'samples_per_type_train', 'samples_per_type_val', 
                        'samples_per_type_test', 'segment_length_sec', 'partition']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Set future prediction time
        future_prediction_sec = config.get('future_prediction_sec', self.future_prediction_sec)
        
        print("Creating future location prediction labeled segments with configuration:")
        print(f"  Segment length: {config['segment_length_sec']} seconds")
        print(f"  Future prediction time: {future_prediction_sec} seconds")
        print(f"  Partitions: {config['partition']}")
        print("  Map filter: de_mirage only")
        print("  Players per segment: 5 (single team, all alive)")
        
        # Get all match IDs and filter for de_mirage
        print("\nDiscovering matches...")
        all_match_ids = self._get_all_match_ids()
        print(f"Found {len(all_match_ids)} total matches")
        
        # Filter for de_mirage matches
        mirage_matches = []
        print("Filtering for de_mirage matches...")
        for match_id in tqdm(all_match_ids, desc="Checking maps"):
            if self._is_mirage_match(match_id):
                mirage_matches.append(match_id)
        
        print(f"Found {len(mirage_matches)} de_mirage matches")
        
        if not mirage_matches:
            print("No de_mirage matches found. Exiting.")
            return
        
        # Extract segments with multiprocessing
        print(f"\nExtracting future location segments using {self.num_processes} processes...")
        print(f"  CPU usage: {self.cpu_usage*100:.0f}% ({self.num_processes}/{mp.cpu_count()} cores)")
        
        # Prepare arguments for worker processes
        worker_args = [
            (match_id, self.data_dir, self.trajectory_folder, 
             config['segment_length_sec'], future_prediction_sec, self.tick_rate)
            for match_id in mirage_matches
        ]
        
        # Process all matches in parallel
        all_segments = []
        print(f"  Processing {len(mirage_matches)} de_mirage matches...")
        try:
            with mp.Pool(processes=self.num_processes) as pool:
                all_results = []
                with tqdm(total=len(worker_args), desc="Processing matches", unit="match") as pbar:
                    for result in pool.imap_unordered(self._process_match_worker, worker_args):
                        all_results.append(result)
                        pbar.update(1)
            
            # Flatten results
            for segments in all_results:
                all_segments.extend(segments)
            
            print(f"  Found {len(all_segments)} valid future location segments")
                
        except Exception as e:
            print(f"Error processing matches: {e}")
            return
        
        if not all_segments:
            print("No valid future location segments found. Exiting.")
            return
        
        # Apply sampling limits
        print("\nApplying sampling limits...")
        partition_config = {}
        for partition in config['partition']:
            key = f'samples_per_type_{partition}'
            if key in config:
                partition_config[partition] = config[key]
        
        # Limit total samples per partition
        total_samples_needed = sum(partition_config.values())
        if len(all_segments) > total_samples_needed:
            # Randomly sample segments
            random.shuffle(all_segments)
            all_segments = all_segments[:total_samples_needed]
            print(f"  Sampled {len(all_segments)} segments from available pool")
        
        # Create output CSV
        print("\nCreating output CSV...")
        df = self._create_future_location_csv(all_segments, config)
        
        print("\nFuture location prediction labeling completed successfully!")


if __name__ == "__main__":
    # Configuration
    if sys.platform == "win32":
        DATA_DIR = fr"C:\Users\wangy\projects\CTFM\data\{DATA_PARTITION}"
        OUTPUT_DIR = fr"C:\Users\wangy\projects\CTFM\data\{DATA_PARTITION}\labels"
        PARTITION_CSV_PATH = fr"C:\Users\wangy\projects\CTFM\data\{DATA_PARTITION}\video_path_partitioned.csv"
        TRAJECTORY_FOLDER = "trajectory"
        VIDEO_FOLDER = "video"
        EXAMPLE_VIDEO_SAVE_PATH = r"C:\Users\wangy\projects\CTFM\example_video"
    else:
        DATA_DIR = f"/project2/ustun_1726/CTFM/data/{DATA_PARTITION}"
        OUTPUT_DIR = f"/project2/ustun_1726/CTFM/data/{DATA_PARTITION}/labels"
        PARTITION_CSV_PATH = f"/project2/ustun_1726/CTFM/data/{DATA_PARTITION}/video_partitioned_chunked_downsized.csv"
        TRAJECTORY_FOLDER = "trajectory"
        VIDEO_FOLDER = "video_544x306"
        EXAMPLE_VIDEO_SAVE_PATH = "/home1/yunzhewa/projects/CTFM/example_video"
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create future location label creator
    future_location_creator = MultiAgentFutureLocationLabelCreator(
        DATA_DIR, 
        OUTPUT_DIR,
        PARTITION_CSV_PATH,
        TRAJECTORY_FOLDER, 
        VIDEO_FOLDER,
        example_video_save_path=EXAMPLE_VIDEO_SAVE_PATH,
        cpu_usage=0.9,
        future_prediction_sec=10  # K = 10 seconds
    )
    
    # Parse future location predictions (production configuration)
    future_location_creator.parse_future_location_predictions({
        'output_file_name': 'multi_agent_future_locations_5s_10s.csv',
        'samples_per_type_train': 5000,   # Total samples for training
        'samples_per_type_val': 1000,     # Total samples for validation  
        'samples_per_type_test': 1000,    # Total samples for testing
        'segment_length_sec': 5,          # 5-second segments
        'future_prediction_sec': 10,      # 10-second future prediction (K = 10)
        'partition': ['train', 'val', 'test']
    })
