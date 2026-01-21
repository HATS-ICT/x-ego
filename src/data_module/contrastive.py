"""
Contrastive Learning DataModule (Stage 1)

Lightning DataModule for team alignment contrastive learning.
Handles variable number of agents per sample.
"""

from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import base class
try:
    from .base import BaseDataModule
except ImportError:
    from base import BaseDataModule

# Import dataset and collate function
try:
    from ..dataset.contrastive import ContrastiveDataset, contrastive_collate_fn
except ImportError:
    from dataset.contrastive import ContrastiveDataset, contrastive_collate_fn


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
        
        # Store contrastive-specific parameters
        self.min_agents = cfg.data.min_agents
    
    def _create_base_dataset(self):
        """Create the base contrastive dataset."""
        return ContrastiveDataset(cfg=self.cfg)
    
    def _get_collate_fn(self):
        """Get the collate function for contrastive learning."""
        return contrastive_collate_fn
    
    def _copy_dataset_attributes(self, base_dataset, partition_dataset):
        """Copy dataset attributes."""
        attrs = ['min_agents', 'video_processor',
                 'cfg', 'data_cfg', 'path_cfg', 'data_root', 
                 'target_fps', 'fixed_duration_seconds', 'time_jitter_max_seconds']
        for attr in attrs:
            if hasattr(base_dataset, attr):
                setattr(partition_dataset, attr, getattr(base_dataset, attr))
    
    def _print_partition_info(self, df, partition_name: str):
        """Print partition information."""
        total_samples = len(df)
        logger.info(f"{partition_name} partition: {total_samples} samples")
        
        if 'pov_team_side' in df.columns:
            team_counts = df['pov_team_side'].value_counts()
            logger.info(f"  Team distribution: {dict(team_counts)}")
    
    def _store_dataset_info(self, base_dataset):
        """Store dataset information."""
        logger.info(f"Full contrastive dataset: {len(base_dataset)} samples")
        logger.info(f"Minimum agents: {self.min_agents}")
    
    def _get_val_drop_last(self) -> bool:
        """Drop last batch in validation for consistent batch sizes."""
        return True
    
    def _get_test_drop_last(self) -> bool:
        """Drop last batch in test for consistent batch sizes."""
        return True
