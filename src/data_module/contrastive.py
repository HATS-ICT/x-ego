"""
Contrastive Learning DataModule (Stage 1)

Lightning DataModule for team alignment contrastive learning.
Handles variable number of agents per sample.
"""

from typing import Dict, Any
from rich import print as rprint

from .base import BaseDataModule
from ..dataset.contrastive import ContrastiveDataset
from ..dataset.collate import contrastive_collate_fn


class ContrastiveDataModule(BaseDataModule):
    """
    Lightning DataModule for contrastive learning (team alignment).
    
    Provides train/validation/test splits for stage 1 contrastive learning.
    Handles variable number of agents per sample.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize Contrastive DataModule.
        
        Args:
            cfg: Configuration dictionary
        """
        super().__init__(cfg)
        
    def _create_base_dataset(self):
        """Create the base contrastive dataset."""
        return ContrastiveDataset(cfg=self.cfg)
    
    def _get_collate_fn(self):
        """Get the collate function for contrastive learning."""
        return contrastive_collate_fn
    
    def _copy_dataset_attributes(self, base_dataset, partition_dataset):
        """Copy dataset attributes."""
        attrs = ['video_processor',
                 'cfg', 'data_cfg', 'path_cfg', 'data_root', 
                 'target_fps', 'fixed_duration_seconds', 'time_jitter_max_seconds']
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
        rprint(f"[green]✓[/green] Full contrastive dataset: [bold]{len(base_dataset):,}[/bold] samples")
    
    def _get_val_drop_last(self) -> bool:
        """Drop last batch in validation for consistent batch sizes."""
        return True
    
    def _get_test_drop_last(self) -> bool:
        """Drop last batch in test for consistent batch sizes."""
        return True
