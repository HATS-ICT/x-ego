"""
Linear Probing DataModule (Stage 2)

Lightning DataModule for downstream task linear probing.
"""

from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .base import BaseDataModule
except ImportError:
    from base import BaseDataModule

try:
    from ..dataset.linear_probe import LinearProbeDataset, linear_probe_collate_fn
except ImportError:
    from dataset.linear_probe import LinearProbeDataset, linear_probe_collate_fn


class LinearProbeDataModule(BaseDataModule):
    """
    Lightning DataModule for Stage 2 linear probing.
    
    Works with any task defined in task_definitions.csv.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize Linear Probe DataModule.
        
        Args:
            cfg: Configuration with task info
        """
        super().__init__(cfg)
        
        self.task_id = cfg.task.task_id
        self.ml_form = cfg.task.ml_form
    
    def _create_base_dataset(self):
        """Create the base linear probe dataset."""
        return LinearProbeDataset(cfg=self.cfg)
    
    def _get_collate_fn(self):
        """Get collate function."""
        return linear_probe_collate_fn
    
    def _copy_dataset_attributes(self, base_dataset, partition_dataset):
        """Copy dataset attributes."""
        attrs = ['task_id', 'ml_form', 'output_dim', 'num_classes', 
                 'label_columns', 'num_agents']
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
        logger.info(f"Task: {base_dataset.task_id}")
        logger.info(f"ML form: {base_dataset.ml_form}")
        logger.info(f"Output dim: {base_dataset.output_dim}")
        logger.info(f"Num classes: {base_dataset.num_classes}")
        logger.info(f"Full dataset: {len(base_dataset)} samples")
