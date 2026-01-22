"""
Downstream Task Dataset (Stage 2)

Dataset for downstream tasks using raw video input (single agent per sample).
Loads video and task-specific labels based on task_id from task_definitions.csv.

No precomputed embeddings - video is processed through the model's encoder.
"""

from pathlib import Path
from typing import Dict
import polars as pl
import torch
from torch.utils.data import Dataset
from rich import print as rprint

from .dataset_utils import (
    init_video_processor,
    construct_video_path,
    load_video_clip,
    transform_video,
)


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
        
        # Load label CSV
        data_root = Path(cfg.path.data)
        label_path = data_root / cfg.data.labels_folder / cfg.data.labels_filename
        
        if not label_path.exists():
            raise FileNotFoundError(f"Labels file not found: {label_path}")
        
        self.df = pl.read_csv(label_path, null_values=[])
        rprint(f"[green]CHECK[/green] Loaded [bold]{len(self.df):,}[/bold] samples from [dim]{label_path}[/dim]")
        
        # Add original CSV index
        self.df = self.df.with_row_index("original_csv_idx")
        
        # Filter by partition
        if cfg.data.partition != 'all':
            initial_count = len(self.df)
            self.df = self.df.filter(pl.col('partition') == cfg.data.partition)
            filtered_count = len(self.df)
            rprint(f"[blue]->[/blue] Filtered dataset from [bold]{initial_count:,}[/bold] to [bold]{filtered_count:,}[/bold] samples for partition [cyan]'{cfg.data.partition}'[/cyan]")
        
        # Initialize video processor
        self._init_video_processor()
        
        # Parse label column configuration
        self._parse_label_columns()
        
        rprint(f"[green]CHECK[/green] Task: [bold]{cfg.task.task_id}[/bold] ([cyan]{cfg.task.ml_form}[/cyan])")
        rprint(f"  Output dim: [bold]{cfg.task.output_dim}[/bold], Num classes: [bold]{cfg.task.num_classes}[/bold]")
    
    def _init_video_processor(self):
        """Initialize video processor based on model type from config."""
        self.video_processor, self.processor_type = init_video_processor(self.cfg)
        self.model_type = self.cfg.model.encoder.video.model_type
    
    def _parse_label_columns(self):
        """Parse label column names from task config."""
        label_column = self.cfg.task.label_column
        
        # Label column can be a single string or semicolon-separated list
        if isinstance(label_column, str):
            if ';' in label_column:
                self.label_columns = label_column.split(';')
            else:
                self.label_columns = [label_column]
        else:
            self.label_columns = list(label_column)
        
        # Verify columns exist
        df_columns = self.df.columns
        for col in self.label_columns:
            if col not in df_columns:
                raise ValueError(f"Label column '{col}' not found in CSV. Available: {list(df_columns)}")
        
        rprint(f"  Label columns: [magenta]{self.label_columns}[/magenta]")
    
    
    def _get_label(self, row: Dict) -> torch.Tensor:
        """
        Extract label from row based on task type.
        
        Returns tensor of appropriate shape for the task.
        """
        ml_form = self.cfg.task.ml_form
        
        if ml_form == 'binary_cls':
            # Single binary value
            value = row[self.label_columns[0]]
            return torch.tensor(float(value), dtype=torch.float32)
        
        elif ml_form == 'multi_cls':
            # Single class index
            value = row[self.label_columns[0]]
            return torch.tensor(int(value), dtype=torch.long)
        
        elif ml_form == 'multi_label_cls':
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
                    num_classes = self.cfg.task.num_classes
                    label = torch.zeros(num_classes, dtype=torch.float32)
                    label[int(value)] = 1.0
                    return label
            else:
                # Multiple columns - each column is a binary indicator
                values = [float(row[col]) for col in self.label_columns]
                return torch.tensor(values, dtype=torch.float32)
        
        elif ml_form == 'regression':
            # Continuous value(s)
            if len(self.label_columns) == 1:
                value = row[self.label_columns[0]]
                return torch.tensor(float(value), dtype=torch.float32)
            else:
                values = [float(row[col]) for col in self.label_columns]
                return torch.tensor(values, dtype=torch.float32)
        
        else:
            raise ValueError(f"Unknown ml_form: {ml_form}")
    
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
        row = self.df.row(idx, named=True)
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
        video_path = construct_video_path(self.cfg, match_id, player_id, round_num)
        video_clip = load_video_clip(self.cfg, video_path, start_seconds, end_seconds)
        video = transform_video(self.video_processor, self.processor_type, video_clip)  # [T, C, H, W]
        
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
    rprint("[dim]DownstreamDataset test placeholder[/dim]")
