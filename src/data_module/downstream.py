"""
Downstream Task DataModule (Stage 2)

Lightning DataModule for downstream task linear probing.
Uses single-agent raw video input.
"""

from typing import Dict, Any
from rich import print as rprint

from .base import BaseDataModule
from ..dataset.downstream import DownstreamDataset
from ..dataset.collate import downstream_collate_fn

class DownstreamDataModule(BaseDataModule):
    """
    Lightning DataModule for Stage 2 downstream linear probing.
    
    Uses single-agent raw video input.
    Works with any task defined in task_definitions.csv.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize Downstream DataModule.
        
        Args:
            cfg: Configuration with task info
        """
        super().__init__(cfg)
    
    def _create_base_dataset(self):
        """Create the base downstream dataset."""
        return DownstreamDataset(cfg=self.cfg)
    
    def _get_collate_fn(self):
        """Get collate function for single-agent data."""
        return downstream_collate_fn
    
    def _copy_dataset_attributes(self, base_dataset, partition_dataset):
        """Copy dataset attributes to partition datasets."""
        attrs = [
            'label_columns', 'video_processor',
            'cfg', 'model_type', 'processor_type',
        ]
        for attr in attrs:
            if hasattr(base_dataset, attr):
                setattr(partition_dataset, attr, getattr(base_dataset, attr))
    
    def _print_partition_info(self, df, partition_name: str):
        """Print partition information."""
        total_samples = len(df)
        rprint(f"[cyan]{partition_name}[/cyan] partition: [bold]{total_samples:,}[/bold] samples")
        
        if 'pov_team_side' in df.columns:
            team_counts = df['pov_team_side'].value_counts()
            team_dict = dict(team_counts)
            rprint(f"  Team distribution: [magenta]{team_dict}[/magenta]")
    
    def _store_dataset_info(self, base_dataset):
        """Store dataset information."""
        rprint(f"[green]OK[/green] Task: [bold]{base_dataset.cfg.task.task_id}[/bold]")
        rprint(f"  ML form: [cyan]{base_dataset.cfg.task.ml_form}[/cyan]")
        rprint(f"  Output dim: [bold]{base_dataset.cfg.task.output_dim}[/bold]")
        rprint(f"  Num classes: [bold]{base_dataset.cfg.task.num_classes}[/bold]")
        rprint(f"  Full dataset: [bold]{len(base_dataset):,}[/bold] samples")
