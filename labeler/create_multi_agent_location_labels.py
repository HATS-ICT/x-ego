import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import random
import multiprocessing as mp
from tqdm import tqdm
import subprocess
import json
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ctfm.labeler.label_creator import SegmentLabelCreator

DATA_PARTITION = "sample"


class MultiAgentLocationLabelCreator(SegmentLabelCreator):
    """
    A specialized label creator for multi-agent location data.
    
    Extracts 5-second segments where all 10 players are alive and records their
    location information (X, Y, Z coordinates, side, place, video paths).
    Filters for de_mirage map only.
    """
    
    def __init__(self, data_dir: str, output_dir: str, partition_csv_path: str,
                 trajectory_folder: str = "trajectory", video_folder: str = "video_544x306", 
                 tick_rate: int = 64, seed: int = 42, example_video_save_path: str = None,
                 cpu_usage: float = 0.9):
        """Initialize MultiAgentLocationLabelCreator with same parameters as parent class."""
        super().__init__(data_dir, output_dir, partition_csv_path, trajectory_folder, 
                        video_folder, tick_rate, seed, example_video_save_path, cpu_usage)
    
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
    
    def _load_all_player_trajectories(self, match_id: str, round_num: int) -> Dict[str, pd.DataFrame]:
        """
        Load trajectory data for all players in a specific match and round.
        
        Args:
            match_id: Match identifier
            round_num: Round number
        
        Returns:
            Dict mapping player steamid to their trajectory DataFrame
        """
        match_dir = Path(self.data_dir) / "recording" / "trajectory" / match_id
        player_trajectories = {}
        
        if not match_dir.exists():
            return player_trajectories
        
        # Get all player directories
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
        
        return player_trajectories
    
    def _find_valid_segments(self, player_trajectories: Dict[str, pd.DataFrame], 
                           segment_length_sec: int, overlap_factor: float = 0.0) -> List[Dict]:
        """
        Find valid 5-second segments where all 10 players are alive.
        
        Args:
            player_trajectories: Dict mapping steamid to trajectory DataFrame
            segment_length_sec: Segment duration in seconds
            overlap_factor: Overlap between segments
        
        Returns:
            List of valid segment dictionaries
        """
        if len(player_trajectories) != 10:
            return []  # Must have exactly 10 players
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        
        # Find common tick range across all players
        all_ticks = []
        for df in player_trajectories.values():
            if not df.empty:
                all_ticks.extend(df['tick'].tolist())
        
        if not all_ticks:
            return []
        
        min_tick = min(all_ticks)
        max_tick = max(all_ticks)
        
        # Calculate step size based on overlap factor
        step_size = int(segment_ticks * (1.0 - overlap_factor))
        if step_size <= 0:
            step_size = 1
        
        # Generate segments
        current_tick = min_tick
        while current_tick + segment_ticks <= max_tick:
            end_tick = current_tick + segment_ticks
            
            # Check if all 10 players have data in this tick range and are alive
            player_data = {}
            all_players_valid = True
            
            for steamid, df in player_trajectories.items():
                # Get data for this tick range
                segment_data = df[(df['tick'] >= current_tick) & (df['tick'] <= end_tick)]
                
                if segment_data.empty:
                    all_players_valid = False
                    break
                
                # Check if player is alive throughout the segment (health > 0)
                if (segment_data['health'] <= 0).any():
                    all_players_valid = False
                    break
                
                # Get representative data (middle of segment)
                mid_idx = len(segment_data) // 2
                mid_data = segment_data.iloc[mid_idx]
                
                player_data[steamid] = {
                    'steamid': steamid,
                    'name': mid_data['name'],
                    'side': mid_data['side'],
                    'X': mid_data['X'],
                    'Y': mid_data['Y'],
                    'Z': mid_data['Z'],
                    'place': mid_data['place'],
                    'map_name': mid_data['map_name'],
                    'round_num': mid_data['round_num']
                }
            
            if all_players_valid and len(player_data) == 10:
                # Sort players by side (T first, then CT) for consistent ordering
                sorted_players = sorted(player_data.values(), key=lambda p: (p['side'], p['steamid']))
                
                # Create segment info
                first_player_df = list(player_trajectories.values())[0]
                min_tick_in_trajectory = first_player_df['tick'].min()
                
                segment_info = {
                    'start_tick': current_tick,
                    'end_tick': end_tick,
                    'start_seconds': current_tick / self.tick_rate,
                    'end_seconds': end_tick / self.tick_rate,
                    'normalized_start_seconds': (current_tick - min_tick_in_trajectory) / self.tick_rate,
                    'normalized_end_seconds': (end_tick - min_tick_in_trajectory) / self.tick_rate,
                    'duration_seconds': segment_length_sec,
                    'map_name': sorted_players[0]['map_name'],
                    'round_num': sorted_players[0]['round_num'],
                    'players': sorted_players
                }
                
                segments.append(segment_info)
            
            current_tick += step_size
        
        return segments
    
    def _extract_match_segments(self, match_id: str, segment_length_sec: int, 
                              overlap_factor: float = 0.0) -> List[Dict]:
        """
        Extract all valid segments from a match across all rounds.
        
        Args:
            match_id: Match identifier
            segment_length_sec: Segment duration in seconds
            overlap_factor: Overlap between segments
        
        Returns:
            List of segment dictionaries
        """
        if not self._is_mirage_match(match_id):
            return []  # Skip non-mirage matches
        
        all_segments = []
        
        # Try rounds 1-16 (typical CS:GO match length)
        for round_num in range(1, 17):
            player_trajectories = self._load_all_player_trajectories(match_id, round_num)
            
            if len(player_trajectories) == 10:  # Only process if all 10 players present
                segments = self._find_valid_segments(player_trajectories, segment_length_sec, overlap_factor)
                
                for segment in segments:
                    segment['match_id'] = match_id
                    all_segments.append(segment)
        
        return all_segments
    
    @staticmethod
    def _process_match_worker(args):
        """Worker function for multiprocessing match extraction."""
        (match_id, data_dir, trajectory_folder, segment_length_sec, overlap_factor, tick_rate) = args
        
        try:
            # Create a temporary instance for processing
            temp_creator = MultiAgentLocationLabelCreator.__new__(MultiAgentLocationLabelCreator)
            temp_creator.data_dir = Path(data_dir)
            temp_creator.trajectory_folder = trajectory_folder
            temp_creator.tick_rate = tick_rate
            temp_creator._is_mirage_match = MultiAgentLocationLabelCreator._is_mirage_match.__get__(temp_creator)
            temp_creator._load_all_player_trajectories = MultiAgentLocationLabelCreator._load_all_player_trajectories.__get__(temp_creator)
            temp_creator._find_valid_segments = MultiAgentLocationLabelCreator._find_valid_segments.__get__(temp_creator)
            
            return temp_creator._extract_match_segments(match_id, segment_length_sec, overlap_factor)
            
        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            return []
    
    def _create_multi_agent_csv(self, all_segments: List[Dict], config: Dict[str, Any]):
        """Create the final CSV output for multi-agent location data."""
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
            
            # Construct video path for partition lookup (use backslashes to match CSV format on Windows)
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
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'start_seconds': segment['start_seconds'],
                'end_seconds': segment['end_seconds'],
                'normalized_start_seconds': segment['normalized_start_seconds'],
                'normalized_end_seconds': segment['normalized_end_seconds'],
                'match_id': match_id,
                'round_num': round_num,
                'map_name': segment['map_name']
            }
            
            # Add data for all 10 players
            for i, player in enumerate(segment['players']):
                steamid = player['steamid']
                if os.name == 'nt':  # Windows
                    video_path = f"data\\{DATA_PARTITION}\\recording\\{self.video_folder}\\{match_id}\\{steamid}\\round_{round_num}.mp4"
                else:  # Unix-like systems
                    video_path = f"data/{DATA_PARTITION}/recording/{self.video_folder}/{match_id}/{steamid}/round_{round_num}.mp4"
                
                row[f'player_{i}_id'] = steamid
                row[f'player_{i}_X'] = player['X']
                row[f'player_{i}_Y'] = player['Y']
                row[f'player_{i}_Z'] = player['Z']
                row[f'player_{i}_side'] = player['side']
                row[f'player_{i}_place'] = player['place']
                row[f'player_{i}_video_path'] = video_path
                row[f'player_{i}_name'] = player['name']
            
            output_rows.append(row)
            idx += 1
        
        # Create DataFrame and sort by match_id, round_num, then partition
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'], ascending=[True, True, True])
            df = df.reset_index(drop=True)
            # Update idx after sorting
            df['idx'] = range(len(df))
        
        output_path = self.output_dir / config['output_file_name']
        df.to_csv(output_path, index=False)
        
        print(f"\nSaved {len(output_rows)} multi-agent location segments to {output_path}")
        
        # Print summary statistics
        if len(df) > 0:
            print("\nSummary by partition:")
            for partition in df['partition'].unique():
                partition_data = df[df['partition'] == partition]
                print(f"  {partition}: {len(partition_data)} segments")
            
            print("\nOverall summary:")
            print(f"  Total segments: {len(df)}")
            print(f"  Matches processed: {df['match_id'].nunique()}")
            print(f"  Rounds covered: {df['round_num'].nunique()}")
        else:
            print("\nNo segments were saved - check partition matching and de_mirage filtering.")
        
        return df
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get the actual duration of a video file using ffmpeg."""
        try:
            cmd = [
                'ffprobe', 
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data['format']['duration'])
                return duration
            else:
                return -1
                
        except Exception as e:
            print(f"Error getting duration for {video_path}: {e}")
            return -1
    
    def _create_multi_agent_video_grid(self, segments: List[Dict], label: str, 
                                     example_video_save_path: str, num_examples: int = 5) -> List[str]:
        """
        Create video grids with 2 rows (T vs CT) and 5 columns for multi-agent visualization.
        
        Args:
            segments: List of segment dictionaries
            label: Label for the video (used in filename)
            example_video_save_path: Directory to save videos
            num_examples: Number of example videos to create
        
        Returns:
            List of created video file paths
        """
        try:
            import subprocess
            import tempfile
            import os
        except ImportError:
            print("Required modules not available for video grid creation")
            return []
        
        created_videos = []
        output_dir = Path(example_video_save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select segments to process
        selected_segments = segments[:num_examples] if len(segments) >= num_examples else segments
        
        for seg_idx, segment in enumerate(selected_segments):
            try:
                # Create temporary directory for individual player videos
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    player_videos = []
                    
                    # Separate players by side
                    t_players = [p for p in segment['players'] if p['side'] == 't']
                    ct_players = [p for p in segment['players'] if p['side'] == 'ct']
                    
                    # Ensure we have 5 players per side
                    if len(t_players) != 5 or len(ct_players) != 5:
                        print(f"Skipping segment with incorrect team sizes: T={len(t_players)}, CT={len(ct_players)}")
                        continue
                    
                    # Process all 10 players (T team first, then CT team)
                    all_players = t_players + ct_players
                    
                    for player_idx, player in enumerate(all_players):
                        steamid = player['steamid']
                        match_id = segment['match_id']
                        round_num = segment['round_num']
                        
                        # Construct full video path
                        video_path = Path(self.data_dir) / f"recording/{self.video_folder}/{match_id}/{steamid}/round_{round_num}.mp4"
                        
                        if not video_path.exists():
                            print(f"Video not found: {video_path}")
                            continue
                        
                        # Create individual player segment with reduced resolution
                        output_video = temp_path / f"player_{player_idx:02d}.mp4"
                        
                        # Extract segment and resize to 360x202 (even height for h264)
                        cmd = [
                            'ffmpeg', '-y',
                            '-ss', str(segment['normalized_start_seconds']),
                            '-i', str(video_path),
                            '-t', str(segment['duration_seconds']),
                            '-vf', 'scale=360:202',
                            '-c:v', 'libx264',
                            '-preset', 'fast',
                            '-crf', '28',
                            '-an',  # Remove audio
                            str(output_video)
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode == 0 and output_video.exists():
                            player_videos.append(str(output_video))
                        else:
                            print(f"Failed to create player video for {steamid}: {result.stderr}")
                    
                    # Create grid if we have all 10 videos
                    if len(player_videos) == 10:
                        # Create 2x5 grid: top row T team, bottom row CT team
                        grid_output = output_dir / f"multi_agent_location_{label}_{seg_idx+1:03d}.mp4"
                        
                        # Build filter_complex for 2x5 grid with correct FFmpeg syntax
                        filter_parts = []
                        
                        # Create top row (T team - players 0-4)
                        top_row_inputs = "".join([f"[{i}:v]" for i in range(5)])
                        filter_parts.append(f"{top_row_inputs}hstack=inputs=5[top]")
                        
                        # Create bottom row (CT team - players 5-9)
                        bottom_row_inputs = "".join([f"[{i}:v]" for i in range(5, 10)])
                        filter_parts.append(f"{bottom_row_inputs}hstack=inputs=5[bottom]")
                        
                        # Stack rows vertically
                        filter_parts.append("[top][bottom]vstack=inputs=2[out]")
                        
                        # Add text overlays for team labels (adjusted for new height)
                        filter_parts.append(
                            "[out]drawtext=text='T TEAM (TOP)':fontsize=24:fontcolor=orange:"
                            "x=10:y=10:box=1:boxcolor=black@0.5[labeled1]"
                        )
                        filter_parts.append(
                            "[labeled1]drawtext=text='CT TEAM (BOTTOM)':fontsize=24:fontcolor=blue:"
                            "x=10:y=212:box=1:boxcolor=black@0.5[final]"
                        )
                        
                        full_filter = ";".join(filter_parts)
                        
                        # Debug: Print the filter string
                        print(f"FFmpeg filter string: {full_filter}")
                        
                        cmd = [
                            'ffmpeg', '-y'
                        ] + [item for video in player_videos for item in ['-i', video]] + [
                            '-filter_complex', full_filter,
                            '-map', '[final]',
                            '-c:v', 'libx264',
                            '-preset', 'fast',
                            '-crf', '23',
                            '-r', '30',
                            str(grid_output)
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode == 0 and grid_output.exists():
                            created_videos.append(str(grid_output))
                            print(f"Created multi-agent video grid: {grid_output}")
                        else:
                            print(f"Failed to create video grid: {result.stderr}")
                    else:
                        print(f"Could not create all player videos, got {len(player_videos)}/10")
                        
            except Exception as e:
                print(f"Error creating video grid for segment {seg_idx}: {e}")
        
        return created_videos
    
    def _create_example_videos_multi_agent(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Create example video grids for multi-agent location data."""
        print("\nCreating multi-agent example video grids...")
        
        # Check if FFmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  Warning: FFmpeg not found. Skipping video creation.")
            return
        
        if not self.example_video_save_path:
            print("  No example video save path specified.")
            return
        
        num_examples = config.get('num_video_example', 5)
        
        # Convert DataFrame back to segment format for video creation
        segments = []
        for _, row in df.iterrows():
            players = []
            for i in range(10):
                if f'player_{i}_id' in row:
                    player = {
                        'steamid': row[f'player_{i}_id'],
                        'name': row[f'player_{i}_name'],
                        'side': row[f'player_{i}_side'],
                        'X': row[f'player_{i}_X'],
                        'Y': row[f'player_{i}_Y'],
                        'Z': row[f'player_{i}_Z'],
                        'place': row[f'player_{i}_place']
                    }
                    players.append(player)
            
            segment = {
                'match_id': row['match_id'],
                'round_num': row['round_num'],
                'normalized_start_seconds': row['normalized_start_seconds'],
                'duration_seconds': row['seg_duration_sec'],
                'players': players
            }
            segments.append(segment)
        
        # Create video grids
        created_videos = self._create_multi_agent_video_grid(
            segments, 'mirage_locations', self.example_video_save_path, num_examples
        )
        
        print(f"  Created {len(created_videos)} multi-agent video grids in: {self.example_video_save_path}")
    
    def parse_multi_agent_locations(self, config: Dict[str, Any]):
        """
        Parse multi-agent location data and create labeled segments.
        
        Args:
            config: Configuration dictionary with keys:
                - output_file_name: Name of the output CSV file
                - samples_per_type_train/val/test: Max samples for each partition
                - segment_length_sec: Length of segments in seconds
                - partition: List of partitions to include ['train', 'val', 'test']
                - overlap_factor: Overlap between segments (optional, default: 0.0)
                - save_example_label_video: Whether to save example videos (optional)
                - num_video_example: Number of example videos to create (optional, default: 5)
        """
        # Validate configuration
        required_keys = ['output_file_name', 'samples_per_type_train', 'samples_per_type_val', 
                        'samples_per_type_test', 'segment_length_sec', 'partition']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        print("Creating multi-agent location labeled segments with configuration:")
        print(f"  Segment length: {config['segment_length_sec']} seconds")
        print(f"  Overlap factor: {config.get('overlap_factor', 0.0)} ({config.get('overlap_factor', 0.0)*100:.0f}% overlap)")
        print(f"  Partitions: {config['partition']}")
        print("  Map filter: de_mirage only")
        print("  Players per segment: 10 (all alive)")
        
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
        print(f"\nExtracting multi-agent segments using {self.num_processes} processes...")
        print(f"  CPU usage: {self.cpu_usage*100:.0f}% ({self.num_processes}/{mp.cpu_count()} cores)")
        
        # Prepare arguments for worker processes
        worker_args = [
            (match_id, self.data_dir, self.trajectory_folder, 
             config['segment_length_sec'], config.get('overlap_factor', 0.0), self.tick_rate)
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
            
            print(f"  Found {len(all_segments)} valid multi-agent segments")
                
        except Exception as e:
            print(f"Error processing matches: {e}")
            return
        
        if not all_segments:
            print("No valid multi-agent segments found. Exiting.")
            return
        
        # Apply sampling limits
        print("\nApplying sampling limits...")
        partition_config = {}
        for partition in config['partition']:
            key = f'samples_per_type_{partition}'
            if key in config:
                partition_config[partition] = config[key]
        
        # Since we don't have traditional "labels", we'll just limit total samples per partition
        total_samples_needed = sum(partition_config.values())
        if len(all_segments) > total_samples_needed:
            # Randomly sample segments
            random.shuffle(all_segments)
            all_segments = all_segments[:total_samples_needed]
            print(f"  Sampled {len(all_segments)} segments from available pool")
        
        # Create output CSV
        print("\nCreating output CSV...")
        df = self._create_multi_agent_csv(all_segments, config)
        
        # Create example videos if requested
        if config.get('save_example_label_video', False) and self.example_video_save_path and len(df) > 0:
            self._create_example_videos_multi_agent(df, config)
        
        print("\nMulti-agent location labeling completed successfully!")


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
    
    # Create multi-agent location label creator
    multi_agent_creator = MultiAgentLocationLabelCreator(
        DATA_DIR, 
        OUTPUT_DIR,
        PARTITION_CSV_PATH,
        TRAJECTORY_FOLDER, 
        VIDEO_FOLDER,
        example_video_save_path=EXAMPLE_VIDEO_SAVE_PATH,
        cpu_usage=0.9
    )
    
    # Parse multi-agent locations (production configuration)
    multi_agent_creator.parse_multi_agent_locations({
        'output_file_name': 'multi_agent_locations_5s.csv',
        'samples_per_type_train': 5000,   # Total samples for training
        'samples_per_type_val': 1000,     # Total samples for validation  
        'samples_per_type_test': 1000,    # Total samples for testing
        'segment_length_sec': 5,         # 5-second segments
        'overlap_factor': 0.3,           # 30% overlap
        'partition': ['train', 'val', 'test'],
        'save_example_label_video': False,
        'num_video_example': 5
    })
