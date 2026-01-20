"""
Base class for task label creation.

Extends the LocationPredictionBase pattern from labeler/base.py with
additional utilities for diverse prediction tasks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
import multiprocessing as mp
from tqdm import tqdm
from abc import ABC, abstractmethod
import json


class TaskCreatorBase(ABC):
    """
    Base class for creating task labels.
    
    Provides common functionality for:
    - Loading trajectory data
    - Loading event data (kills, damages, bomb, shots)
    - Loading metadata
    - Segment extraction with configurable stride
    - Player filtering by side
    """
    
    def __init__(self, data_dir: str, output_dir: str, partition_csv_path: str,
                 trajectory_folder: str = "trajectory", 
                 event_folder: str = "event",
                 metadata_folder: str = "metadata",
                 video_folder: str = "video_544x306_30fps", 
                 tick_rate: int = 64, seed: int = 42, 
                 cpu_usage: float = 0.9, stride_sec: float = 1.0):
        """
        Initialize the TaskCreatorBase.
        
        Args:
            data_dir: Path to the data directory
            output_dir: Directory where output CSV files will be saved
            partition_csv_path: Path to the match round partition CSV file
            trajectory_folder: Name of trajectory folder
            event_folder: Name of event folder
            metadata_folder: Name of metadata folder
            video_folder: Name of video folder
            tick_rate: Game tick rate (default: 64)
            seed: Random seed for reproducibility
            cpu_usage: Fraction of CPU cores to use
            stride_sec: Step size in seconds between segments
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.partition_csv_path = partition_csv_path
        self.trajectory_folder = trajectory_folder
        self.event_folder = event_folder
        self.metadata_folder = metadata_folder
        self.video_folder = video_folder
        self.tick_rate = tick_rate
        self.seed = seed
        self.cpu_usage = cpu_usage
        
        if stride_sec <= 0.0:
            raise ValueError("stride_sec must be > 0.0")
        self.stride_sec = stride_sec
        
        self.num_processes = max(1, int(mp.cpu_count() * cpu_usage))
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_partition_data()
        
        self.debug = False
        
        # Cache for loaded data
        self._trajectory_cache = {}
        self._event_cache = {}
        self._metadata_cache = {}
    
    def _load_partition_data(self):
        """Load match round partition data from CSV."""
        partition_path = Path(self.partition_csv_path)
        
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition CSV not found: {partition_path}")
        
        self.partition_df = pd.read_csv(partition_path)
        print(f"Loaded {len(self.partition_df)} match-round entries from partition file")
    
    # ========== Trajectory Loading ==========
    
    def _load_player_trajectories(self, match_id: str, round_num: int) -> Dict[str, pd.DataFrame]:
        """
        Load trajectory data for all players in a specific match and round.
        
        Args:
            match_id: Match identifier
            round_num: Round number
        
        Returns:
            Dict mapping player steamid to their trajectory DataFrame
        """
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
    
    # ========== Event Loading ==========
    
    def _load_event_data(self, match_id: str, event_type: str) -> pd.DataFrame:
        """
        Load event data (kills, damages, bomb, shots, rounds).
        
        Args:
            match_id: Match identifier
            event_type: Type of event file (kills, damages, bomb, shots, rounds)
        
        Returns:
            DataFrame with event data
        """
        cache_key = (match_id, event_type)
        if cache_key in self._event_cache:
            return self._event_cache[cache_key]
        
        event_path = self.data_dir / self.event_folder / match_id / f"{event_type}.csv"
        
        if not event_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(event_path)
            self._event_cache[cache_key] = df
            return df
        except Exception as e:
            print(f"Error loading {event_type} for match {match_id}: {e}")
            return pd.DataFrame()
    
    def _load_kills(self, match_id: str, round_num: Optional[int] = None) -> pd.DataFrame:
        """Load kills data, optionally filtered by round."""
        df = self._load_event_data(match_id, "kills")
        if round_num is not None and not df.empty:
            df = df[df['round_num'] == round_num]
        return df
    
    def _load_damages(self, match_id: str, round_num: Optional[int] = None) -> pd.DataFrame:
        """Load damages data, optionally filtered by round."""
        df = self._load_event_data(match_id, "damages")
        if round_num is not None and not df.empty:
            df = df[df['round_num'] == round_num]
        return df
    
    def _load_bomb(self, match_id: str, round_num: Optional[int] = None) -> pd.DataFrame:
        """Load bomb events data, optionally filtered by round."""
        df = self._load_event_data(match_id, "bomb")
        if round_num is not None and not df.empty:
            df = df[df['round_num'] == round_num]
        return df
    
    def _load_shots(self, match_id: str, round_num: Optional[int] = None) -> pd.DataFrame:
        """Load shots data, optionally filtered by round."""
        df = self._load_event_data(match_id, "shots")
        if round_num is not None and not df.empty:
            df = df[df['round_num'] == round_num]
        return df
    
    def _load_rounds(self, match_id: str) -> pd.DataFrame:
        """Load rounds data for a match."""
        return self._load_event_data(match_id, "rounds")
    
    # ========== Metadata Loading ==========
    
    def _load_metadata(self, match_id: str) -> Dict[str, Any]:
        """
        Load metadata JSON for a match.
        
        Args:
            match_id: Match identifier
        
        Returns:
            Dict with metadata
        """
        if match_id in self._metadata_cache:
            return self._metadata_cache[match_id]
        
        metadata_path = self.data_dir / self.metadata_folder / f"{match_id}.json"
        
        if not metadata_path.exists():
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            self._metadata_cache[match_id] = data
            return data
        except Exception as e:
            print(f"Error loading metadata for match {match_id}: {e}")
            return {}
    
    # ========== Player Filtering ==========
    
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
    
    # ========== Tick Range Utilities ==========
    
    def _find_player_death_tick(self, player_df: pd.DataFrame) -> Optional[int]:
        """Find the tick when a specific player dies, or None if they survive."""
        if player_df.empty or 'health' not in player_df.columns:
            return None
        
        death_mask = player_df['health'] <= 0
        if death_mask.any():
            first_death_idx = death_mask.idxmax()
            return player_df.loc[first_death_idx, 'tick']
        return None
    
    def _find_first_death_tick(self, player_trajectories: Dict[str, pd.DataFrame]) -> int:
        """Find the tick when the first player dies."""
        if not player_trajectories:
            return 0
        
        death_ticks = []
        
        for df in player_trajectories.values():
            if not df.empty and 'health' in df.columns:
                death_mask = df['health'] <= 0
                if death_mask.any():
                    first_death_idx = death_mask.idxmax()
                    death_tick = df.loc[first_death_idx, 'tick']
                    death_ticks.append(death_tick)
        
        if death_ticks:
            return min(death_ticks)
        else:
            all_ticks = []
            for df in player_trajectories.values():
                if not df.empty:
                    all_ticks.extend(df['tick'].tolist())
            return max(all_ticks) if all_ticks else 0
    
    def _get_valid_tick_range(self, player_trajectories: Dict[str, pd.DataFrame]) -> Tuple[int, int]:
        """
        Get the valid tick range where all players have data and are alive.
        
        Returns:
            Tuple of (min_tick, max_tick_before_death)
        """
        if not player_trajectories:
            return 0, 0
        
        all_ticks = []
        for df in player_trajectories.values():
            if not df.empty:
                all_ticks.extend(df['tick'].tolist())
        
        if not all_ticks:
            return 0, 0
        
        min_tick = min(all_ticks)
        max_tick = max(all_ticks)
        
        first_death_tick = self._find_first_death_tick(player_trajectories)
        max_tick_alive = min(max_tick, first_death_tick)
        
        return min_tick, max_tick_alive
    
    # ========== Data Extraction ==========
    
    def _extract_player_data_at_tick(self, df: pd.DataFrame, target_tick: int) -> Optional[Dict]:
        """
        Extract player data at a specific tick.
        
        Args:
            df: Player trajectory DataFrame
            target_tick: Target tick
        
        Returns:
            Dict with player data or None if not found
        """
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
    
    def _check_event_in_window(self, events_df: pd.DataFrame, 
                               start_tick: int, end_tick: int) -> bool:
        """Check if any event occurs within the tick window."""
        if events_df.empty:
            return False
        return ((events_df['tick'] >= start_tick) & (events_df['tick'] <= end_tick)).any()
    
    def _get_events_in_window(self, events_df: pd.DataFrame,
                              start_tick: int, end_tick: int) -> pd.DataFrame:
        """Get all events within the tick window."""
        if events_df.empty:
            return pd.DataFrame()
        return events_df[(events_df['tick'] >= start_tick) & (events_df['tick'] <= end_tick)]
    
    # ========== Spatial Utilities ==========
    
    def _compute_team_centroid(self, players_data: List[Dict]) -> Tuple[float, float, float]:
        """Compute team centroid from player positions."""
        if not players_data:
            return 0.0, 0.0, 0.0
        
        xs = [p['X_norm'] for p in players_data if p.get('X_norm') is not None]
        ys = [p['Y_norm'] for p in players_data if p.get('Y_norm') is not None]
        zs = [p['Z_norm'] for p in players_data if p.get('Z_norm') is not None]
        
        if not xs:
            return 0.0, 0.0, 0.0
        
        return np.mean(xs), np.mean(ys), np.mean(zs)
    
    def _compute_team_spread(self, players_data: List[Dict]) -> float:
        """Compute team spatial spread (std of positions)."""
        if len(players_data) < 2:
            return 0.0
        
        xs = [p['X_norm'] for p in players_data if p.get('X_norm') is not None]
        ys = [p['Y_norm'] for p in players_data if p.get('Y_norm') is not None]
        
        if len(xs) < 2:
            return 0.0
        
        # Use 2D spread (X, Y)
        return np.sqrt(np.std(xs)**2 + np.std(ys)**2)
    
    def _compute_distance(self, p1: Dict, p2: Dict) -> float:
        """Compute Euclidean distance between two players."""
        if not p1 or not p2:
            return float('inf')
        
        x1, y1, z1 = p1.get('X_norm', 0), p1.get('Y_norm', 0), p1.get('Z_norm', 0)
        x2, y2, z2 = p2.get('X_norm', 0), p2.get('Y_norm', 0), p2.get('Z_norm', 0)
        
        return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    
    def _compute_movement_direction(self, prev_pos: Dict, curr_pos: Dict) -> int:
        """
        Compute movement direction index (0-8).
        
        Returns index in [N, NE, E, SE, S, SW, W, NW, STATIONARY]
        """
        if not prev_pos or not curr_pos:
            return 8  # STATIONARY
        
        dx = curr_pos.get('X_norm', 0) - prev_pos.get('X_norm', 0)
        dy = curr_pos.get('Y_norm', 0) - prev_pos.get('Y_norm', 0)
        
        # Check if stationary (threshold)
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return 8  # STATIONARY
        
        # Compute angle and map to direction
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Map angle to 8 directions (N=90°, E=0°, S=-90°, W=180°)
        # Adjust for game coordinates where Y typically increases northward
        if -22.5 <= angle < 22.5:
            return 2  # E
        elif 22.5 <= angle < 67.5:
            return 1  # NE
        elif 67.5 <= angle < 112.5:
            return 0  # N
        elif 112.5 <= angle < 157.5:
            return 7  # NW
        elif angle >= 157.5 or angle < -157.5:
            return 6  # W
        elif -157.5 <= angle < -112.5:
            return 5  # SW
        elif -112.5 <= angle < -67.5:
            return 4  # S
        else:  # -67.5 <= angle < -22.5
            return 3  # SE
    
    def _count_alive_players(self, players_data: List[Dict]) -> int:
        """Count number of alive players."""
        return sum(1 for p in players_data if p.get('health', 0) > 0)
    
    # ========== Video Path ==========
    
    def _construct_video_path(self, match_id: str, steamid: str, round_num: int) -> str:
        """Construct video path for a player's round."""
        return str((Path("data") / self.video_folder / match_id / steamid / f"round_{round_num}.mp4"))
    
    # ========== Abstract Methods ==========
    
    @abstractmethod
    def _extract_segments_from_round(self, match_id: str, round_num: int, 
                                     config: Dict[str, Any]) -> List[Dict]:
        """
        Extract segments from a specific round. Must be implemented by subclasses.
        
        Args:
            match_id: Match identifier
            round_num: Round number
            config: Configuration dictionary
        
        Returns:
            List of segment dictionaries
        """
        pass
    
    @abstractmethod
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Create the final CSV output. Must be implemented by subclasses.
        
        Args:
            all_segments: List of all extracted segments
            config: Configuration dictionary
        
        Returns:
            DataFrame with the created CSV data
        """
        pass
    
    # ========== Balanced Sampling ==========
    
    def _detect_label_field(self, segments: List[Dict]) -> tuple:
        """
        Auto-detect the label field and task type from segment dictionaries.
        
        Returns:
            Tuple of (label_field, task_type) where task_type is one of:
            'binary', 'multi_cls', 'multi_label', 'regression', 'unknown'
        """
        if not segments:
            return None, 'unknown'
        
        sample = segments[0]
        
        # Check for common binary classification labels
        binary_fields = ['has_kill', 'pov_dies', 'pov_kills', 'in_combat', 
                        'bomb_planted', 'bomb_site', 'will_plant', 
                        'post_plant_outcome', 'round_winner']
        for field in binary_fields:
            if field in sample:
                val = sample[field]
                if isinstance(val, (int, float, np.integer, np.floating)) and val in [0, 1, 0.0, 1.0]:
                    return field, 'binary'
        
        # Check for multi-class labels (integer values)
        multi_cls_fields = ['movement_dir', 'place_idx', 'alive_count', 
                           'round_outcome', 'weapon_idx', 'direction']
        for field in multi_cls_fields:
            if field in sample:
                val = sample[field]
                if isinstance(val, (int, np.integer)):
                    return field, 'multi_cls'
        
        # Check for multi-label classification (list of labels)
        if 'place_labels' in sample:
            return 'place_labels', 'multi_label'
        if 'direction_labels' in sample:
            return 'direction_labels', 'multi_label'
        
        # Check for regression labels
        regression_fields = ['team_spread', 'centroid_x', 'centroid_y', 
                            'nearest_distance', 'speed', 'speed_0']
        for field in regression_fields:
            if field in sample:
                return field, 'regression'
        
        return None, 'unknown'
    
    def _balanced_downsample_binary(self, segments: List[Dict], max_samples: int,
                                    label_field: str) -> List[Dict]:
        """
        Balanced downsample for binary classification tasks.
        Ensures equal representation of both classes.
        """
        # Group by label
        groups = {0: [], 1: []}
        for seg in segments:
            label = int(seg[label_field])
            groups[label].append(seg)
        
        count_0, count_1 = len(groups[0]), len(groups[1])
        
        # Calculate target samples per class
        samples_per_class = max_samples // 2
        remainder = max_samples % 2
        
        random.seed(self.seed)
        balanced = []
        
        # Sample from each class
        for i, (label, group) in enumerate(sorted(groups.items())):
            target = samples_per_class + (1 if i < remainder else 0)
            if len(group) >= target:
                balanced.extend(random.sample(group, target))
            else:
                # Undersample: take all from minority class, reduce majority
                balanced.extend(group)
                print(f"  Warning: Class {label} has only {len(group)} samples (target: {target})")
        
        random.shuffle(balanced)
        
        # Report balance
        final_0 = sum(1 for s in balanced if int(s[label_field]) == 0)
        final_1 = len(balanced) - final_0
        print(f"  Binary balanced downsample: {len(segments)} -> {len(balanced)}")
        print(f"  Original: Class 0={count_0}, Class 1={count_1} (ratio: {max(count_0, count_1)/max(1, min(count_0, count_1)):.2f}:1)")
        print(f"  Final: Class 0={final_0}, Class 1={final_1} (ratio: {max(final_0, final_1)/max(1, min(final_0, final_1)):.2f}:1)")
        
        return balanced
    
    def _balanced_downsample_multi_cls(self, segments: List[Dict], max_samples: int,
                                       label_field: str) -> List[Dict]:
        """
        Balanced downsample for multi-class classification tasks.
        Uses stratified sampling across all classes.
        """
        # Group by label
        groups = {}
        for seg in segments:
            label = seg[label_field]
            if label not in groups:
                groups[label] = []
            groups[label].append(seg)
        
        num_classes = len(groups)
        if num_classes == 0:
            return segments
        
        # Calculate target samples per class
        samples_per_class = max_samples // num_classes
        remainder = max_samples % num_classes
        
        random.seed(self.seed)
        balanced = []
        
        class_counts_before = {k: len(v) for k, v in groups.items()}
        
        for i, (label, group) in enumerate(sorted(groups.items())):
            target = samples_per_class + (1 if i < remainder else 0)
            if len(group) >= target:
                balanced.extend(random.sample(group, target))
            else:
                balanced.extend(group)
                print(f"  Warning: Class {label} has only {len(group)} samples (target: {target})")
        
        random.shuffle(balanced)
        
        # Report balance
        class_counts_after = {}
        for s in balanced:
            label = s[label_field]
            class_counts_after[label] = class_counts_after.get(label, 0) + 1
        
        max_before = max(class_counts_before.values())
        min_before = min(class_counts_before.values())
        max_after = max(class_counts_after.values()) if class_counts_after else 0
        min_after = min(class_counts_after.values()) if class_counts_after else 0
        
        print(f"  Multi-class balanced downsample: {len(segments)} -> {len(balanced)}")
        print(f"  Classes: {num_classes}, Original imbalance: {max_before/max(1, min_before):.2f}:1")
        print(f"  Final imbalance: {max_after/max(1, min_after):.2f}:1")
        
        return balanced
    
    def _balanced_downsample_multi_label(self, segments: List[Dict], max_samples: int,
                                         label_field: str) -> List[Dict]:
        """
        Balanced downsample for multi-label classification tasks.
        Uses label-wise stratification to ensure each label has good representation.
        """
        if len(segments) <= max_samples:
            return segments
        
        # Convert labels to numpy for easier manipulation
        labels = np.array([seg[label_field] for seg in segments])  # Shape: (N, num_labels)
        num_labels = labels.shape[1]
        
        # Calculate positive rate for each label
        label_positive_rates = labels.sum(axis=0) / len(labels)
        
        # Compute per-sample rarity score (samples with rare labels get higher scores)
        # Use inverse frequency weighting
        label_weights = 1.0 / (label_positive_rates + 0.01)  # Add small epsilon to avoid div by zero
        sample_scores = (labels * label_weights).sum(axis=1)  # Higher = more rare labels
        
        # Select samples using weighted probability based on rarity score
        # This ensures samples with rare labels are more likely to be selected
        probs = sample_scores / sample_scores.sum()
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Sample with probability proportional to rarity score
        indices = np.random.choice(len(segments), size=min(max_samples, len(segments)), 
                                   replace=False, p=probs)
        
        balanced = [segments[i] for i in indices]
        
        # Report balance
        new_labels = np.array([seg[label_field] for seg in balanced])
        new_rates = new_labels.sum(axis=0) / len(new_labels)
        
        print(f"  Multi-label balanced downsample: {len(segments)} -> {len(balanced)}")
        print(f"  Labels: {num_labels}")
        print(f"  Original positive rates: min={label_positive_rates.min()*100:.1f}%, max={label_positive_rates.max()*100:.1f}%")
        print(f"  Final positive rates: min={new_rates.min()*100:.1f}%, max={new_rates.max()*100:.1f}%")
        
        return balanced
    
    def _balanced_downsample(self, segments: List[Dict], max_samples: int, 
                            label_field: str = None) -> List[Dict]:
        """
        Downsample segments while maintaining class balance.
        Auto-detects task type and applies appropriate balancing strategy.
        
        Args:
            segments: List of segment dictionaries
            max_samples: Target number of samples
            label_field: Field name containing the label (auto-detected if None)
        
        Returns:
            Downsampled list of segments with balanced classes
        """
        if len(segments) <= max_samples:
            return segments
        
        # Auto-detect label field and task type if not specified
        detected_field, task_type = self._detect_label_field(segments)
        
        if label_field is None:
            label_field = detected_field
        
        print(f"  Detected task type: {task_type}, label field: {label_field}")
        
        # Apply task-specific balancing
        if task_type == 'binary' and label_field:
            return self._balanced_downsample_binary(segments, max_samples, label_field)
        
        elif task_type == 'multi_cls' and label_field:
            return self._balanced_downsample_multi_cls(segments, max_samples, label_field)
        
        elif task_type == 'multi_label' and label_field:
            return self._balanced_downsample_multi_label(segments, max_samples, label_field)
        
        elif task_type == 'regression':
            # For regression, just random sample
            print(f"  Regression task: random sampling {len(segments)} -> {max_samples}")
            random.seed(self.seed)
            return random.sample(segments, max_samples)
        
        else:
            # Unknown task type, fall back to random sampling
            print(f"  Unknown task type, falling back to random sampling")
            random.seed(self.seed)
            return random.sample(segments, max_samples)
    
    # ========== Main Processing ==========
    
    def process_segments(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Main method to process segments and create labeled data.
        
        Args:
            config: Configuration dictionary with keys:
                - output_file_name: Name of the output CSV file
                - segment_length_sec: Length of segments in seconds
                - partition: List of partitions to include ['train', 'val', 'test']
                - oversample_multiplier: Multiplier for oversampling before balanced downsample (default: 3)
                - Additional parameters specific to each task
        
        Returns:
            DataFrame with processed data
        """
        required_keys = ['output_file_name', 'segment_length_sec', 'partition']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        max_samples = config.get('max_samples')
        oversample_mult = config.get('oversample_multiplier', 3)
        # Collect Nx target for diversity, then downsample with balancing
        collection_target = max_samples * oversample_mult if max_samples else None
        
        print("Processing segments with configuration:")
        print(f"  Segment length: {config['segment_length_sec']} seconds")
        print(f"  Partitions: {config['partition']}")
        if max_samples:
            print(f"  Max samples: {max_samples} (collecting up to {collection_target} with {oversample_mult}x, then balanced downsample)")
        
        filtered_partition_df = self.partition_df[
            self.partition_df['split'].isin(config['partition'])
        ]
        
        print(f"Found {len(filtered_partition_df)} match-round combinations in desired partitions")
        
        if filtered_partition_df.empty:
            print("No match-round combinations found for specified partitions. Exiting.")
            return pd.DataFrame()
        
        # Shuffle match-rounds for diversity when using max_samples
        if max_samples:
            filtered_partition_df = filtered_partition_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            print("  Shuffled match-rounds for diversity")
        
        print("\nExtracting segments...")
        
        all_segments = []
        early_stopped = False
        
        for _, row in tqdm(filtered_partition_df.iterrows(), 
                          total=len(filtered_partition_df), 
                          desc="Processing match-rounds"):
            match_id = row['match_id']
            round_num = row['round_number']
            partition = row['split']
            
            try:
                segments = self._extract_segments_from_round(match_id, round_num, config)
                
                for segment in segments:
                    segment['partition'] = partition
                    segment['match_id'] = match_id
                    segment['round_num'] = round_num
                
                all_segments.extend(segments)
                
                # Early stopping if collection target reached
                if collection_target and len(all_segments) >= collection_target:
                    early_stopped = True
                    break
                
            except Exception as e:
                print(f"Error processing match {match_id}, round {round_num}: {e}")
                continue
        
        if early_stopped:
            print(f"Early stopped at {len(all_segments)} segments (collection_target={collection_target})")
        else:
            print(f"Extracted {len(all_segments)} total segments")
        
        if not all_segments:
            print("No valid segments found. Exiting.")
            return pd.DataFrame()
        
        # Downsample if we collected more than max_samples (with balancing)
        if max_samples and len(all_segments) > max_samples:
            all_segments = self._balanced_downsample(all_segments, max_samples)
        
        print("\nCreating output CSV...")
        df = self._create_output_csv(all_segments, config)
        
        output_path = self.output_dir / config['output_file_name']
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(df)} segments to {output_path}")
        
        if len(df) > 0:
            print("\nSummary by partition:")
            for partition in df['partition'].unique():
                partition_data = df[df['partition'] == partition]
                print(f"  {partition}: {len(partition_data)} segments")
        
        return df
    
    def clear_cache(self):
        """Clear all cached data."""
        self._trajectory_cache.clear()
        self._event_cache.clear()
        self._metadata_cache.clear()
