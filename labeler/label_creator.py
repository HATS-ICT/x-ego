import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import random
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ctfm.labeler.label_data_preview import create_example_videos_for_label, check_ffmpeg_availability


def extract_basic_segment_info(df: pd.DataFrame, start_tick: int, end_tick: int) -> Dict[str, Any]:
    """
    Extract basic segment information from trajectory data.
    
    Args:
        df: Trajectory DataFrame
        start_tick: Start tick of segment
        end_tick: End tick of segment
    
    Returns:
        Dictionary with basic segment info (mainly player_name)
    """
    segment = df[(df['tick'] >= start_tick) & (df['tick'] <= end_tick)]
    
    if segment.empty:
        return {}
    
    # Get player name from the first row
    player_name = segment.iloc[0]['name'] if 'name' in segment.columns else 'unknown'
    
    return {
        'player_name': player_name
    }


class SegmentLabelCreator:
    """
    A flexible class for creating labeled segments from Counter-Strike trajectory data.
    
    Features:
    - Multiprocessing support for fast trajectory processing (default: 90% CPU usage)
    - Flexible labeling with single or multiple columns
    - Overlap support for segment generation
    - Example video creation for label verification
    - Efficient random sampling and balancing across partitions
    """
    
    def __init__(self, data_dir: str, output_dir: str, partition_csv_path: str,
                 trajectory_folder: str = "trajectory", video_folder: str = "video_544x306", 
                 tick_rate: int = 64, seed: int = 42, example_video_save_path: str = None,
                 cpu_usage: float = 0.9):
        """
        Initialize the SegmentLabelCreator.
        
        Args:
            data_dir: Path to the data directory containing trajectory and video files
            output_dir: Directory where output CSV files will be saved
            partition_csv_path: Path to the video partition CSV file
            trajectory_folder: Name of trajectory folder (default: "trajectory")
            video_folder: Name of video folder (default: "video_544x306")
            tick_rate: Game tick rate (default: 64)
            seed: Random seed for reproducibility (default: 42)
            example_video_save_path: Path to save example videos (optional)
            cpu_usage: Fraction of CPU cores to use for multiprocessing (default: 0.9)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.partition_csv_path = partition_csv_path
        self.trajectory_folder = trajectory_folder
        self.video_folder = video_folder
        self.tick_rate = tick_rate
        self.seed = seed
        self.example_video_save_path = example_video_save_path
        self.cpu_usage = cpu_usage
        
        # Calculate number of processes to use
        self.num_processes = max(1, int(mp.cpu_count() * cpu_usage))
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Load video partition data
        self._load_video_partitions()
    
    def _load_video_partitions(self):
        """Load video partition data from CSV."""
        csv_path = Path(self.partition_csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Video partition CSV not found: {csv_path}")
        
        self.video_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.video_df)} video entries from partition file")
    
    def _get_videos_for_partitions(self, partitions: List[str]) -> List[str]:
        """Get video paths for specified partitions (train/val/test)."""
        # Filter for recording type and specified partitions
        filtered_df = self.video_df[
            (self.video_df['split'].isin(partitions)) & 
            (self.video_df['type'] == 'recording')
        ]
        
        print(f"Found {len(filtered_df)} recording videos across partitions: {partitions}")
        return filtered_df['video_path'].tolist()
    
    def _video_path_to_trajectory_path(self, video_path: str) -> str:
        """Convert video path to corresponding trajectory CSV path."""
        path_obj = Path(video_path)
        
        # Handle both absolute and relative paths
        if path_obj.parts[0] == 'data' and path_obj.parts[1] == 'full':
            relative_parts = path_obj.parts[2:]  # Remove 'data/full' prefix
        else:
            relative_parts = path_obj.parts
        
        # Convert video -> trajectory and .mp4 -> .csv
        trajectory_parts = list(relative_parts)
        trajectory_parts[1] = self.trajectory_folder  # recording/video -> recording/trajectory
        trajectory_path = self.data_dir / Path(*trajectory_parts)
        trajectory_path = trajectory_path.with_suffix('.csv')
        
        return str(trajectory_path)
    
    def _trajectory_path_to_video_path(self, trajectory_path: str) -> str:
        """Convert trajectory path to corresponding video path."""
        # Convert from absolute path to relative path format used in partition CSV
        path_obj = Path(trajectory_path)
        
        # Find the 'data/full' part in the path
        path_parts = list(path_obj.parts)
        try:
            # Find where 'data' and 'full' appear in the path
            data_idx = None
            for i, part in enumerate(path_parts):
                if part == 'data' and i + 1 < len(path_parts) and path_parts[i + 1] == 'full':
                    data_idx = i
                    break
            
            if data_idx is not None:
                # Take from 'data' onwards and convert trajectory -> video
                relative_parts = path_parts[data_idx:]
                relative_parts[3] = self.video_folder  # recording/trajectory -> recording/video
                video_path = "/".join(relative_parts).replace(".csv", ".mp4")
                return video_path
        except (IndexError, ValueError):
            pass
        
        # Fallback: simple replacement
        video_path = trajectory_path.replace(self.trajectory_folder, self.video_folder).replace(".csv", ".mp4")
        return video_path
    
    def _load_trajectory_data(self, trajectory_path: str) -> pd.DataFrame:
        """Load trajectory CSV data."""
        if not Path(trajectory_path).exists():
            return pd.DataFrame()  # Return empty DataFrame if file doesn't exist
        
        try:
            return pd.read_csv(trajectory_path, keep_default_na=False)
        except Exception as e:
            print(f"Error loading trajectory {trajectory_path}: {e}")
            return pd.DataFrame()
    
    def _get_consistency_score(self, df: pd.DataFrame, start_tick: int, end_tick: int, 
                              column_names: List[str], target_values: Union[str, Tuple], 
                              per_type_filter: Dict = None) -> float:
        """
        Calculate consistency score for specified columns and values in a segment.
        
        Args:
            df: Trajectory DataFrame
            start_tick: Start tick of segment
            end_tick: End tick of segment
            column_names: List of column names to check
            target_values: Target value(s) - string for single column, tuple for multiple columns
            per_type_filter: Optional dict mapping target values to additional column conditions
        
        Returns:
            Fraction of frames that match the target values and any additional filters
        """
        segment = df[(df['tick'] >= start_tick) & (df['tick'] <= end_tick)]
        if segment.empty:
            return 0.0
        
        if len(column_names) == 1:
            # Single column case
            matches = (segment[column_names[0]] == target_values)
            
            # Apply per_type_filter if specified for this target value
            if per_type_filter and target_values in per_type_filter:
                filter_conditions = per_type_filter[target_values]
                for filter_col, filter_val in filter_conditions.items():
                    matches &= (segment[filter_col] == filter_val)
            
            matching_frames = matches.sum()
        else:
            # Multiple columns case
            if not isinstance(target_values, (tuple, list)) or len(target_values) != len(column_names):
                return 0.0
            
            # Check all columns match their respective target values
            matches = True
            for i, col in enumerate(column_names):
                matches &= (segment[col] == target_values[i])
            
            # Apply per_type_filter if specified for this target value tuple
            if per_type_filter and target_values in per_type_filter:
                filter_conditions = per_type_filter[target_values]
                for filter_col, filter_val in filter_conditions.items():
                    matches &= (segment[filter_col] == filter_val)
            
            matching_frames = matches.sum()
        
        total_frames = len(segment)
        return matching_frames / total_frames
    
    def _extract_segments_from_trajectory(self, trajectory_path: str, segment_length_sec: int,
                                         column_names: List[str], options: List[Union[str, Tuple]],
                                         tolerance: float, overlap_factor: float = 0.0, 
                                         per_type_filter: Dict = None) -> List[Dict]:
        """
        Extract valid segments from a trajectory file.
        
        Args:
            trajectory_path: Path to trajectory CSV file
            segment_length_sec: Segment duration in seconds
            column_names: Columns to check for labeling
            options: Valid label options to match against
            tolerance: Minimum consistency score (0.0-1.0)
            overlap_factor: Overlap between segments (0.0=no overlap, 0.5=50% overlap)
        
        Returns:
            List of valid segment dictionaries
        """
        df = self._load_trajectory_data(trajectory_path)
        if df.empty:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        
        # Get trajectory bounds
        min_tick = df['tick'].min()
        max_tick = df['tick'].max()
        total_ticks = max_tick - min_tick
        
        if total_ticks < segment_ticks:
            return []  # Trajectory too short
        
        # Calculate step size based on overlap factor
        # overlap_factor=0.0 -> step_size = segment_ticks (no overlap)
        # overlap_factor=0.5 -> step_size = segment_ticks/2 (50% overlap)
        step_size = int(segment_ticks * (1.0 - overlap_factor))
        if step_size <= 0:
            step_size = 1  # Minimum step size
        
        # Generate segments with specified overlap
        current_tick = min_tick
        while current_tick + segment_ticks <= max_tick:
            end_tick = current_tick + segment_ticks
            
            # Extract basic segment info
            variables = extract_basic_segment_info(df, current_tick, end_tick)
            
            if not variables:
                current_tick += segment_ticks
                continue
            
            # Check each option to see if this segment matches
            for option in options:
                consistency_score = self._get_consistency_score(
                    df, current_tick, end_tick, column_names, option, per_type_filter
                )
                
                if consistency_score >= tolerance:
                    # Create label string
                    if len(column_names) == 1:
                        label = str(option).lower().replace(' ', '_')
                        label_values = {column_names[0]: option}
                    else:
                        label = "_".join(str(v) for v in option).lower().replace(' ', '_')
                        label_values = {col: val for col, val in zip(column_names, option)}
                    
                    # Get the minimum tick for this trajectory to calculate normalized times
                    min_tick_in_trajectory = df['tick'].min()
                    normalized_start_seconds = (current_tick - min_tick_in_trajectory) / self.tick_rate
                    normalized_end_seconds = (end_tick - min_tick_in_trajectory) / self.tick_rate
                    
                    segment_info = {
                        'trajectory_path': trajectory_path,
                        'start_tick': current_tick,
                        'end_tick': end_tick,
                        'start_seconds': current_tick / self.tick_rate,
                        'end_seconds': end_tick / self.tick_rate,
                        'normalized_start_seconds': normalized_start_seconds,  # For video seeking
                        'normalized_end_seconds': normalized_end_seconds,      # For video seeking
                        'duration_seconds': segment_length_sec,
                        'label': label,
                        'consistency_score': consistency_score,
                        'player_name': variables.get('player_name', 'unknown'),
                        **label_values,  # Add individual column values
                        **variables  # Add all extracted variables
                    }
                    segments.append(segment_info)
                    break  # Only match one option per segment
            
            # Move to next segment with specified overlap
            current_tick += step_size
        
        return segments
    
    @staticmethod
    def _process_video_worker(args):
        """
        Worker function for multiprocessing trajectory extraction.
        
        Args:
            args: Tuple containing (video_path, data_dir, trajectory_folder, video_folder, 
                                  segment_length_sec, column_names, options, tolerance, 
                                  overlap_factor, tick_rate, per_type_filter)
        
        Returns:
            List of segments extracted from the trajectory
        """
        (video_path, data_dir, trajectory_folder, video_folder, 
         segment_length_sec, column_names, options, tolerance, 
         overlap_factor, tick_rate, per_type_filter) = args
        
        try:
            # Convert video path to trajectory path
            path_obj = Path(video_path)
            if path_obj.parts[0] == 'data' and path_obj.parts[1] == 'full':
                relative_parts = path_obj.parts[2:]  # Remove 'data/full' prefix
            else:
                relative_parts = path_obj.parts
            
            # Convert video -> trajectory and .mp4 -> .csv
            trajectory_parts = list(relative_parts)
            trajectory_parts[1] = trajectory_folder  # recording/video -> recording/trajectory
            trajectory_path = Path(data_dir) / Path(*trajectory_parts)
            trajectory_path = trajectory_path.with_suffix('.csv')
            
            # Load trajectory data
            if not trajectory_path.exists():
                return []
            
            try:
                df = pd.read_csv(trajectory_path, keep_default_na=False)
            except Exception as e:
                print(f"Error loading trajectory {trajectory_path}: {e}")
                return []
            
            if df.empty:
                return []
            
            # Extract segments using the same logic as _extract_segments_from_trajectory
            segments = []
            segment_ticks = segment_length_sec * tick_rate
            
            # Get trajectory bounds
            min_tick = df['tick'].min()
            max_tick = df['tick'].max()
            total_ticks = max_tick - min_tick
            
            if total_ticks < segment_ticks:
                return []  # Trajectory too short
            
            # Calculate step size based on overlap factor
            step_size = int(segment_ticks * (1.0 - overlap_factor))
            if step_size <= 0:
                step_size = 1  # Minimum step size
            
            # Generate segments with specified overlap
            current_tick = min_tick
            while current_tick + segment_ticks <= max_tick:
                end_tick = current_tick + segment_ticks
                
                # Extract basic segment info
                variables = extract_basic_segment_info(df, current_tick, end_tick)
                
                if not variables:
                    current_tick += segment_ticks
                    continue
                
                # Check each option to see if this segment matches
                for option in options:
                    # Calculate consistency score
                    segment = df[(df['tick'] >= current_tick) & (df['tick'] <= end_tick)]
                    if segment.empty:
                        continue
                    
                    if len(column_names) == 1:
                        # Single column case
                        matches = (segment[column_names[0]] == option)
                        
                        # Apply per_type_filter if specified for this target value
                        if per_type_filter and option in per_type_filter:
                            filter_conditions = per_type_filter[option]
                            for filter_col, filter_val in filter_conditions.items():
                                matches &= (segment[filter_col] == filter_val)
                        
                        matching_frames = matches.sum()
                    else:
                        # Multiple columns case
                        if not isinstance(option, (tuple, list)) or len(option) != len(column_names):
                            continue
                        
                        # Check all columns match their respective target values
                        matches = True
                        for i, col in enumerate(column_names):
                            matches &= (segment[col] == option[i])
                        
                        # Apply per_type_filter if specified for this target value tuple
                        if per_type_filter and option in per_type_filter:
                            filter_conditions = per_type_filter[option]
                            for filter_col, filter_val in filter_conditions.items():
                                matches &= (segment[filter_col] == filter_val)
                        
                        matching_frames = matches.sum()
                    
                    total_frames = len(segment)
                    consistency_score = matching_frames / total_frames
                    
                    if consistency_score >= tolerance:
                        # Create label string
                        if len(column_names) == 1:
                            label = str(option).lower().replace(' ', '_')
                            label_values = {column_names[0]: option}
                        else:
                            label = "_".join(str(v) for v in option).lower().replace(' ', '_')
                            label_values = {col: val for col, val in zip(column_names, option)}
                        
                        # Get the minimum tick for this trajectory to calculate normalized times
                        min_tick_in_trajectory = df['tick'].min()
                        normalized_start_seconds = (current_tick - min_tick_in_trajectory) / tick_rate
                        normalized_end_seconds = (end_tick - min_tick_in_trajectory) / tick_rate
                        
                        segment_info = {
                            'trajectory_path': str(trajectory_path),
                            'start_tick': current_tick,
                            'end_tick': end_tick,
                            'start_seconds': current_tick / tick_rate,
                            'end_seconds': end_tick / tick_rate,
                            'normalized_start_seconds': normalized_start_seconds,  # For video seeking
                            'normalized_end_seconds': normalized_end_seconds,      # For video seeking
                            'duration_seconds': segment_length_sec,
                            'label': label,
                            'consistency_score': consistency_score,
                            'player_name': variables.get('player_name', 'unknown'),
                            **label_values,  # Add individual column values
                            **variables  # Add all extracted variables
                        }
                        segments.append(segment_info)
                        break  # Only match one option per segment
                
                # Move to next segment with specified overlap
                current_tick += step_size
            
            return segments
            
        except Exception as e:
            # Return empty list on error, with error info for debugging
            print(f"Error processing video {video_path}: {e}")
            return []
    
    def _filter_labels_by_min_samples(self, segments: List[Dict], min_sample_required: int) -> List[Dict]:
        """
        Filter out labels that don't have enough samples across all partitions.
        
        Args:
            segments: List of segment dictionaries
            min_sample_required: Minimum number of samples required per label
        
        Returns:
            Filtered list of segments with only labels that meet minimum requirements
        """
        if min_sample_required <= 0:
            return segments
        
        # Count samples per label
        label_counts = defaultdict(int)
        for segment in segments:
            label_counts[segment['label']] += 1
        
        # Find labels that meet minimum requirement
        valid_labels = set()
        filtered_labels = set()
        
        for label, count in label_counts.items():
            if count >= min_sample_required:
                valid_labels.add(label)
            else:
                filtered_labels.add(label)
                print(f"  Filtering out label '{label}': {count} samples < {min_sample_required} required")
        
        if filtered_labels:
            print(f"  Removed {len(filtered_labels)} labels due to insufficient samples")
            print(f"  Remaining labels: {len(valid_labels)}")
        else:
            print(f"  All {len(valid_labels)} labels meet minimum sample requirement")
        
        # Filter segments to only include valid labels
        filtered_segments = [seg for seg in segments if seg['label'] in valid_labels]
        print(f"  Segments before filtering: {len(segments)}")
        print(f"  Segments after filtering: {len(filtered_segments)}")
        
        return filtered_segments

    def _balance_samples_by_partition(self, segments: List[Dict], partition_config: Dict[str, int]) -> Dict[str, List[Dict]]:
        """
        Balance samples across partitions and labels.
        
        Args:
            segments: List of segment dictionaries
            partition_config: Dict mapping partition names to max samples per label (-1 means use all samples)
        
        Returns:
            Dict mapping partition names to balanced segment lists
        """
        # Get partition info for each segment by looking up video path
        segments_with_partition = []
        unmatched_segments = 0
        
        for segment in segments:
            video_path = self._trajectory_path_to_video_path(segment['trajectory_path'])
            
            # Find partition for this video
            video_match = self.video_df[self.video_df['video_path'] == video_path]
            if not video_match.empty:
                partition = video_match.iloc[0]['split']
                segment_copy = segment.copy()
                segment_copy['partition'] = partition
                segments_with_partition.append(segment_copy)
            else:
                unmatched_segments += 1
                if unmatched_segments <= 5:  # Only print first few mismatches
                    print(f"  Warning: Could not find partition for video: {video_path}")
        
        if unmatched_segments > 0:
            print(f"  Total unmatched segments: {unmatched_segments}/{len(segments)}")
        
        print(f"  Segments with partition info: {len(segments_with_partition)}")
        
        # Group by partition and label
        partition_label_groups = defaultdict(lambda: defaultdict(list))
        partition_counts = defaultdict(int)
        for segment in segments_with_partition:
            partition = segment['partition']
            label = segment['label']
            partition_label_groups[partition][label].append(segment)
            partition_counts[partition] += 1
        
        # Debug: Print partition distribution
        print("  Partition distribution:")
        for partition, count in partition_counts.items():
            print(f"    {partition}: {count} segments")
        
        # Balance each partition
        balanced_partitions = {}
        
        for partition, max_samples in partition_config.items():
            if partition not in partition_label_groups:
                balanced_partitions[partition] = []
                continue
            
            partition_segments = []
            label_groups = partition_label_groups[partition]
            
            print(f"\n{partition.upper()} partition label distribution:")
            for label, label_segments in label_groups.items():
                print(f"  {label}: {len(label_segments)} segments")
                # Sample up to max_samples segments per label, or use all if max_samples is -1
                if max_samples == -1:
                    sampled = label_segments
                    print(f"    Using all {len(sampled)} samples (samples_per_type=-1)")
                else:
                    sampled = random.sample(label_segments, min(max_samples, len(label_segments)))
                    print(f"    Sampled {len(sampled)} samples")
                partition_segments.extend(sampled)
            
            balanced_partitions[partition] = partition_segments
            print(f"  Total {partition} segments: {len(partition_segments)}")
        
        return balanced_partitions
    
    def _create_output_csv(self, balanced_partitions: Dict[str, List[Dict]], 
                          column_names: List[str], config: Dict[str, Any]):
        """Create the final CSV output."""
        output_rows = []
        idx = 0
        
        for partition, segments in balanced_partitions.items():
            for segment in segments:
                # Convert trajectory path to video path
                video_path = self._trajectory_path_to_video_path(segment['trajectory_path'])
                
                row = {
                    'idx': idx,
                    'partition': partition,
                    'seg_duration_sec': segment['duration_seconds'],
                    'start_tick': segment['start_tick'],
                    'end_tick': segment['end_tick'],
                    'start_seconds': segment['start_seconds'],
                    'end_seconds': segment['end_seconds'],
                    'normalized_start_seconds': segment.get('normalized_start_seconds', segment['start_seconds']),
                    'normalized_end_seconds': segment.get('normalized_end_seconds', segment['end_seconds']),
                    'label_columns': "_".join(column_names),
                    'label': segment['label'],
                    'consistency_score': segment['consistency_score'],
                    'video_path': video_path,
                    'player_name': segment['player_name']
                }
                
                # Add individual column values
                for col in column_names:
                    if col in segment:
                        row[f'label_{col}'] = segment[col]
                
                output_rows.append(row)
                idx += 1
        
        # Create DataFrame and save
        df = pd.DataFrame(output_rows)
        output_path = self.output_dir / config['output_file_name']
        df.to_csv(output_path, index=False)
        
        print(f"\nSaved {len(output_rows)} labeled segments to {output_path}")
        
        # Print summary statistics
        if len(df) > 0:
            print("\nSummary by partition:")
            for partition in balanced_partitions.keys():
                partition_data = df[df['partition'] == partition]
                print(f"  {partition}: {len(partition_data)} segments")
                if len(partition_data) > 0:
                    label_counts = partition_data['label'].value_counts()
                    for label, count in label_counts.items():
                        print(f"    {label}: {count}")
            
            print("\nOverall summary:")
            print(f"  Total segments: {len(df)}")
            label_counts = df['label'].value_counts()
            for label, count in label_counts.items():
                print(f"  {label}: {count}")
        else:
            print("\nNo segments were saved - check balancing logic and video path matching.")
    
    def _create_example_videos(self, balanced_partitions: Dict[str, List[Dict]], config: Dict[str, Any]):
        """Create example videos for each label."""
        print("\nCreating example videos...")
        
        # Check if FFmpeg is available
        if not check_ffmpeg_availability():
            print("  Warning: FFmpeg not found. Skipping video creation.")
            print("  Please install FFmpeg to enable video preview functionality.")
            return
        
        num_examples = config.get('num_example_video_per_label', 2)
        
        # Collect all segments by label across all partitions
        segments_by_label = defaultdict(list)
        for partition, segments in balanced_partitions.items():
            for segment in segments:
                label = segment['label']
                segments_by_label[label].append(segment)
        
        total_created = 0
        for label, segments in segments_by_label.items():
            print(f"  Creating {min(num_examples, len(segments))} example videos for label: {label}")
            
            created_videos = create_example_videos_for_label(
                segments=segments,
                label=label,
                example_video_save_path=self.example_video_save_path,
                num_examples=num_examples,
                trajectory_to_video_converter=self._trajectory_path_to_video_path
            )
            
            total_created += len(created_videos)
        
        print(f"  Created {total_created} example videos in: {self.example_video_save_path}")
    
    def _create_example_videos_from_csv(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Create example videos using data from the CSV file."""
        print("\nCreating example videos...")
        
        # Check if FFmpeg is available
        if not check_ffmpeg_availability():
            print("  Warning: FFmpeg not found. Skipping video creation.")
            print("  Please install FFmpeg to enable video preview functionality.")
            return
        
        num_examples = config.get('num_example_video_per_label', 2)
        
        # Group segments by label
        segments_by_label = defaultdict(list)
        for _, row in df.iterrows():
            label = row['label']
            segment = {
                'video_path': row['video_path'],
                'start_seconds': row.get('normalized_start_seconds', row['start_seconds']),
                'end_seconds': row.get('normalized_end_seconds', row['end_seconds']),
                'consistency_score': row['consistency_score'],
                'player_name': row['player_name']
            }
            segments_by_label[label].append(segment)
        
        total_created = 0
        for label, segments in segments_by_label.items():
            print(f"  Creating {min(num_examples, len(segments))} example videos for label: {label}")
            
            created_videos = create_example_videos_for_label(
                segments=segments,
                label=label,
                example_video_save_path=self.example_video_save_path,
                num_examples=num_examples
            )
            
            total_created += len(created_videos)
        
        print(f"  Created {total_created} example videos in: {self.example_video_save_path}")
    
    def parse(self, config: Dict[str, Any]):
        """
        Parse trajectory data and create labeled segments based on configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - output_file_name: Name of the output CSV file
                - samples_per_type_train: Max samples per label for training (use -1 for all samples)
                - samples_per_type_val: Max samples per label for validation (use -1 for all samples)
                - samples_per_type_test: Max samples per label for testing (use -1 for all samples)
                - segment_length_sec: Length of segments in seconds
                - partition: List of partitions to include ['train', 'val', 'test']
                - column_names: List of column names to use for labeling
                - options: List of valid label values (strings for single column, tuples for multiple)
                - tolerance: Minimum consistency score (0.0-1.0) for segment to be valid
                - overlap_factor: Overlap between segments (0.0=no overlap, 0.5=50% overlap, optional)
                - per_type_filter: Dict mapping label values to additional column conditions (optional)
                  Example: {"AWP": {"is_scoped": False}, "SSG 08": {"is_scoped": False}}
                - min_sample_required: Minimum number of samples required per label to be included (optional, default: 0)
                - save_example_label_video: Whether to save example videos (optional, default: False)
                - num_example_video_per_label: Number of example videos per label (optional, default: 2)
        """
        # Validate configuration
        required_keys = ['output_file_name', 'samples_per_type_train', 'samples_per_type_val', 
                        'samples_per_type_test', 'segment_length_sec', 'partition', 
                        'column_names', 'options', 'tolerance']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        print("Creating labeled segments with configuration:")
        print(f"  Segment length: {config['segment_length_sec']} seconds")
        print(f"  Overlap factor: {config.get('overlap_factor', 0.0)} ({config.get('overlap_factor', 0.0)*100:.0f}% overlap)")
        print(f"  Partitions: {config['partition']}")
        print(f"  Label columns: {config['column_names']}")
        print(f"  Options: {len(config['options'])} labels")
        print(f"  Tolerance: {config['tolerance']}")
        if config.get('min_sample_required', 0) > 0:
            print(f"  Min sample required per label: {config['min_sample_required']}")
        
        # Get videos for specified partitions and shuffle for diversity
        video_paths = self._get_videos_for_partitions(config['partition'])
        random.shuffle(video_paths)  # Shuffle for better diversity
        
        # Calculate target samples needed
        partition_config = {}
        total_target_samples = 0
        has_unlimited = False
        for partition in config['partition']:
            key = f'samples_per_type_{partition}'
            if key in config:
                partition_config[partition] = config[key]
                # Estimate total samples needed (options * samples_per_type)
                if config[key] == -1:
                    has_unlimited = True
                else:
                    total_target_samples += config[key] * len(config['options'])
        
        if has_unlimited:
            total_target_samples = -1  # Indicate unlimited
        
        if total_target_samples == -1:
            print("Target total samples: ALL (samples_per_type=-1 specified)")
        else:
            print(f"Target total samples: ~{total_target_samples}")
        
        # Extract segments with multiprocessing
        print(f"\nExtracting segments using {self.num_processes} processes...")
        print(f"  CPU usage: {self.cpu_usage*100:.0f}% ({self.num_processes}/{mp.cpu_count()} cores)")
        all_segments = []
        
        # Prepare arguments for worker processes - process all videos at once
        worker_args = [
            (video_path, self.data_dir, self.trajectory_folder, self.video_folder,
             config['segment_length_sec'], config['column_names'], config['options'],
             config['tolerance'], config.get('overlap_factor', 0.0), self.tick_rate,
             config.get('per_type_filter', None))
            for video_path in video_paths
        ]
        
        # Process all videos in parallel with progress bar
        print(f"  Processing all {len(video_paths)} videos...")
        try:
            with mp.Pool(processes=self.num_processes) as pool:
                # Use imap_unordered with tqdm for progress tracking
                all_results = []
                with tqdm(total=len(worker_args), desc="Processing videos", unit="video") as pbar:
                    for result in pool.imap_unordered(self._process_video_worker, worker_args):
                        all_results.append(result)
                        pbar.update(1)
            
            # Flatten results
            for segments in all_results:
                all_segments.extend(segments)
            
            print(f"  Found {len(all_segments)} segments total")
                
        except Exception as e:
            print(f"Error processing videos: {e}")
            return
        
        print(f"Found {len(all_segments)} valid segments total")
        
        if not all_segments:
            print("No valid segments found. Exiting.")
            return
        
        # Filter labels by minimum sample requirement
        min_sample_required = config.get('min_sample_required', 0)
        if min_sample_required > 0:
            print(f"\nFiltering labels with minimum sample requirement: {min_sample_required}")
            all_segments = self._filter_labels_by_min_samples(all_segments, min_sample_required)
            
            if not all_segments:
                print("No segments remaining after filtering. Exiting.")
                return
        
        # Balance samples across partitions
        print("\nBalancing samples across partitions...")
        print(f"Raw segments before balancing: {len(all_segments)}")
        balanced_partitions = self._balance_samples_by_partition(all_segments, partition_config)
        total_balanced = sum(len(segs) for segs in balanced_partitions.values())
        print(f"Balanced segments after balancing: {total_balanced}")
        
        # Create output CSV
        print("\nCreating output CSV...")
        self._create_output_csv(balanced_partitions, config['column_names'], config)
        
        # Create example videos if requested
        if config.get('save_example_label_video', False) and self.example_video_save_path:
            # Read the CSV to get segments with video_path
            output_path = self.output_dir / config['output_file_name']
            df = pd.read_csv(output_path)
            self._create_example_videos_from_csv(df, config)
        
        print("\nSegment labeling completed successfully!")


if __name__ == "__main__":
    # Example usage
    DATA_DIR = r"C:\Users\wangy\projects\CTFM\data\full"
    OUTPUT_DIR = r"C:\Users\wangy\projects\CTFM\data\full\labels"
    PARTITION_CSV_PATH = r"C:\Users\wangy\projects\CTFM\data\full\video_path_partitioned.csv"
    TRAJECTORY_FOLDER = "trajectory"
    VIDEO_FOLDER = "video"  # This should match the folder name in partition CSV
    EXAMPLE_VIDEO_SAVE_PATH = r"C:\Users\wangy\projects\CTFM\example_video"
    
    segment_label_creator = SegmentLabelCreator(
        DATA_DIR, 
        OUTPUT_DIR,
        PARTITION_CSV_PATH,
        TRAJECTORY_FOLDER, 
        VIDEO_FOLDER,
        example_video_save_path=EXAMPLE_VIDEO_SAVE_PATH,
        cpu_usage=0.9  # Use 90% of available CPU cores for multiprocessing
    )
    
    from ctfm.labeler.consts import WEAPON_OPTIONS
    # Example: Weapon classification with per_type_filter
    segment_label_creator.parse({
        'output_file_name': 'weapon_5s.csv',
        'samples_per_type_train': 10000,
        'samples_per_type_val': 2000,
        'samples_per_type_test': 2000,
        'segment_length_sec': 5,
        'overlap_factor': 0.5,  # 50% overlap
        'partition': ['train', 'val', 'test'],
        'column_names': ['active_weapon_name'],
        'options': WEAPON_OPTIONS,
        'tolerance': 0.9,
        'per_type_filter': {
            'AWP': {'is_scoped': False},      # AWP segments only when not scoped
            'SSG 08': {'is_scoped': False}    # SSG 08 segments only when not scoped
        },
        'save_example_label_video': True,
        'num_example_video_per_label': 3
    })
    
    # segment_label_creator.parse({
    #     'output_file_name': 'map_place_5s.csv',
    #     'samples_per_type_train': 10,
    #     'samples_per_type_val': 10,
    #     'samples_per_type_test': 10,
    #     'segment_length_sec': 5,
    #     'overlap_factor': 0.5,  # 50% overlap
    #     'partition': ['train', 'val', 'test'],
    #     'column_names': ['map_name', 'place'],
    #     'options': [
    #         ("de_ancient", "Alley"),
    #         ("de_ancient", "BombsiteA"),
    #         ("de_ancient", "Water"),
    #         ("de_anubis", "Alley"),
    #         ("de_anubis", "BackofB"),
    #         ("de_anubis", "SnipersNest"),
    #         ("de_anubis", "Street"),
    #         ("de_anubis", "TSideUpper")
    #     ],
    #     'tolerance': 0.7,
    #     'save_example_label_video': True,
    #     'num_example_video_per_label': 2
    # })

    