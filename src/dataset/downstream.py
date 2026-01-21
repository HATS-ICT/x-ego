"""
Downstream Task Dataset (Stage 2)

Dataset for downstream tasks using raw video input (single agent per sample).
Loads video and task-specific labels based on task_id from task_definitions.csv.

No precomputed embeddings - video is processed through the model's encoder.
"""

import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoVideoProcessor
from decord import VideoReader, cpu

try:
    from utils.dataset_utils import apply_minimap_mask
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_utils import apply_minimap_mask

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def downstream_collate_fn(batch):
    """
    Collate function for downstream dataset.
    
    Args:
        batch: List of dictionaries containing:
            - 'video': Video tensor [T, C, H, W]
            - 'label': Task-specific label (shape depends on task)
            - metadata fields
    
    Returns:
        Dictionary with batched tensors
    """
    collated = {}
    
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key in ['pov_team_side', 'match_id', 'player_id']:
            # Keep string values as lists
            collated[key] = values
        else:
            # Stack tensors
            collated[key] = torch.utils.data.default_collate(values)
    
    return collated


class DownstreamDataset(Dataset):
    """
    Dataset for Stage 2 downstream linear probing.
    
    Loads:
    - Raw video for a single agent per sample
    - Task-specific labels from CSV
    
    Works with any task defined in task_definitions.csv.
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize Downstream Dataset.
        
        Args:
            cfg: Configuration with:
                - task.task_id: Task identifier
                - task.ml_form: ML form (binary_cls, multi_cls, etc.)
                - task.label_column: Column name(s) for labels
                - data.labels_filename: Path to labels CSV
        """
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.task_cfg = cfg.task
        
        # Task info
        self.task_id = self.task_cfg.task_id
        self.ml_form = self.task_cfg.ml_form
        self.output_dim = self.task_cfg.output_dim
        self.num_classes = self.task_cfg.num_classes
        
        # Video parameters
        self.target_fps = self.data_cfg.target_fps
        self.fixed_duration_seconds = self.data_cfg.fixed_duration_seconds
        self.mask_minimap = self.data_cfg.mask_minimap
        self.time_jitter_max_seconds = self.data_cfg.time_jitter_max_seconds
        
        # Path configuration
        self.data_root = Path(cfg.path.data)
        
        # Load label CSV
        label_path = self.data_root / self.data_cfg.labels_folder / self.data_cfg.labels_filename
        
        if not label_path.exists():
            raise FileNotFoundError(f"Labels file not found: {label_path}")
        
        self.df = pd.read_csv(label_path, keep_default_na=False)
        logger.info(f"Loaded {len(self.df)} samples from {label_path}")
        
        # Store original CSV index
        self.df['original_csv_idx'] = self.df.index
        
        # Filter by partition
        partition = self.data_cfg.partition
        if partition != 'all':
            initial_count = len(self.df)
            self.df = self.df[self.df['partition'] == partition].reset_index(drop=True)
            logger.info(f"Filtered from {initial_count} to {len(self.df)} samples for partition '{partition}'")
        
        # Initialize video processor
        self._init_video_processor()
        
        # Parse label column configuration
        self._parse_label_columns()
        
        logger.info(f"Task: {self.task_id} ({self.ml_form})")
        logger.info(f"Output dim: {self.output_dim}, Num classes: {self.num_classes}")
    
    def _init_video_processor(self):
        """Initialize video processor based on configuration."""
        processor_model = self.data_cfg.video_processor_model
        
        if self.data_cfg.video_size_mode == "resize_center_crop":
            self.video_processor = AutoVideoProcessor.from_pretrained(processor_model)
        elif self.data_cfg.video_size_mode == "resize_distort":
            self.video_processor = AutoVideoProcessor.from_pretrained(
                processor_model,
                do_center_crop=False
            )
            self.video_processor.size = {"width": 224, "height": 224}
        else:
            raise ValueError(f"Unsupported video_size_mode: {self.data_cfg.video_size_mode}")
    
    def _parse_label_columns(self):
        """Parse label column names from task config."""
        label_column = self.task_cfg.label_column
        
        # Label column can be a single string or semicolon-separated list
        if isinstance(label_column, str):
            if ';' in label_column:
                self.label_columns = label_column.split(';')
            else:
                self.label_columns = [label_column]
        else:
            self.label_columns = list(label_column)
        
        # Verify columns exist
        for col in self.label_columns:
            if col not in self.df.columns:
                raise ValueError(f"Label column '{col}' not found in CSV. Available: {list(self.df.columns)}")
        
        logger.info(f"Label columns: {self.label_columns}")
    
    def _to_absolute_path(self, path: str) -> str:
        """Convert relative path to absolute path using configured data directory."""
        if not Path(path).is_absolute():
            path_obj = Path(path)
            if path_obj.parts[0] == "data":
                relative_path = Path(*path_obj.parts[1:])
                path = str(self.data_root / relative_path)
            else:
                path = str(self.data_root / path)
        return path
    
    def _construct_video_path(self, match_id: str, player_id: str, round_num: int) -> str:
        """Construct video path for a player's round."""
        video_folder = self.cfg.data.video_folder
        video_path = Path('data') / video_folder / str(match_id) / str(player_id) / f"round_{round_num}.mp4"
        return str(video_path)
    
    def _load_video_clip(self, video_path: str, start_seconds: float, end_seconds: float) -> torch.Tensor:
        """
        Load video clip using decord.
        
        Args:
            video_path: Path to the video file
            start_seconds: Start time of the clip
            end_seconds: End time of the clip
            
        Returns:
            Video tensor of shape (num_frames, channels, height, width)
        """
        expected_frames = int(self.fixed_duration_seconds * self.target_fps)
        video_full_path = self._to_absolute_path(video_path)
        
        try:
            decoder = VideoReader(video_full_path, ctx=cpu(0))
            video_fps = decoder.get_avg_fps()
            
            # Sample frames at target_fps
            timestamps = np.linspace(
                start_seconds, 
                start_seconds + self.fixed_duration_seconds, 
                expected_frames, 
                endpoint=False
            )
            
            # Apply time jitter if configured
            if self.time_jitter_max_seconds > 0:
                jitter = np.random.uniform(
                    -self.time_jitter_max_seconds, 
                    self.time_jitter_max_seconds, 
                    size=len(timestamps)
                )
                timestamps = timestamps + jitter
                total_duration = len(decoder) / video_fps
                timestamps = np.clip(timestamps, 0, total_duration)
            
            frame_indices = (timestamps * video_fps).astype(int)
            max_frame_index = len(decoder) - 1
            frame_indices = np.clip(frame_indices, 0, max_frame_index)
            
            video_clip = decoder.get_batch(frame_indices.tolist())
            video_clip = torch.from_numpy(video_clip.asnumpy()).permute(0, 3, 1, 2).half()
            
            if self.mask_minimap:
                video_clip = apply_minimap_mask(video_clip)
            
            return video_clip
        except Exception as e:
            logger.warning(f"Failed to load video {video_path}: {e}, using placeholder")
            return torch.zeros(expected_frames, 3, 306, 544, dtype=torch.float16)
    
    def _transform_video(self, video_clip: torch.Tensor) -> torch.Tensor:
        """Transform video clip using the video processor."""
        video_processed = self.video_processor(video_clip, return_tensors="pt")
        video_features = video_processed.pixel_values_videos.squeeze(0)
        return video_features
    
    def _get_label(self, row: pd.Series) -> torch.Tensor:
        """
        Extract label from row based on task type.
        
        Returns tensor of appropriate shape for the task.
        """
        if self.ml_form == 'binary_cls':
            # Single binary value
            value = row[self.label_columns[0]]
            return torch.tensor(float(value), dtype=torch.float32)
        
        elif self.ml_form == 'multi_cls':
            # Single class index
            value = row[self.label_columns[0]]
            return torch.tensor(int(value), dtype=torch.long)
        
        elif self.ml_form == 'multi_label_cls':
            # Multi-hot vector - label column should contain the multi-hot encoding
            # or we need to construct it from multiple columns
            if len(self.label_columns) == 1:
                # Single column with multi-hot or needs parsing
                col = self.label_columns[0]
                value = row[col]
                
                # Check if it's already a list/array stored as string
                if isinstance(value, str) and value.startswith('['):
                    import ast
                    value = ast.literal_eval(value)
                    return torch.tensor(value, dtype=torch.float32)
                else:
                    # Assume it's a single value to be one-hot encoded
                    label = torch.zeros(self.num_classes, dtype=torch.float32)
                    label[int(value)] = 1.0
                    return label
            else:
                # Multiple columns - each column is a binary indicator
                values = [float(row[col]) for col in self.label_columns]
                return torch.tensor(values, dtype=torch.float32)
        
        elif self.ml_form == 'regression':
            # Continuous value(s)
            if len(self.label_columns) == 1:
                value = row[self.label_columns[0]]
                return torch.tensor(float(value), dtype=torch.float32)
            else:
                values = [float(row[col]) for col in self.label_columns]
                return torch.tensor(values, dtype=torch.float32)
        
        else:
            raise ValueError(f"Unknown ml_form: {self.ml_form}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            dict: Dictionary containing:
                - 'video': Video tensor [T, C, H, W]
                - 'label': Task-specific label
                - metadata fields
        """
        row = self.df.iloc[idx]
        original_csv_idx = row['original_csv_idx']
        
        # Extract video metadata
        # Convert ticks to seconds (tick_rate = 64)
        tick_rate = 64
        start_seconds = row['start_tick'] / tick_rate
        end_seconds = row['end_tick'] / tick_rate
        match_id = row['match_id']
        round_num = row['round_num']
        
        # Get player ID
        player_id = row['pov_steamid']
        
        # Construct video path and load video
        video_path = self._construct_video_path(match_id, player_id, round_num)
        video_clip = self._load_video_clip(video_path, start_seconds, end_seconds)
        video = self._transform_video(video_clip)  # [T, C, H, W]
        
        # Get label
        label = self._get_label(row)
        
        # Get team side (CSV uses 'pov_side')
        pov_team_side = row.get('pov_side', 'unknown')
        if isinstance(pov_team_side, str):
            pov_team_side = pov_team_side.upper()
        pov_team_side_encoded = 1 if pov_team_side == 'CT' else 0
        
        return {
            'video': video,
            'label': label,
            'pov_team_side': pov_team_side,
            'pov_team_side_encoded': torch.tensor(pov_team_side_encoded, dtype=torch.long),
            'original_csv_idx': original_csv_idx,
            'match_id': str(match_id),
            'player_id': str(player_id),
        }


if __name__ == "__main__":
    print("DownstreamDataset test placeholder")
