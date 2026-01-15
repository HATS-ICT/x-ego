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
    from ..dataset.teammate_location_nowcast import TeammateLocationNowcastDataset, teammate_location_nowcast_collate_fn
except ImportError:
    from dataset.teammate_location_nowcast import TeammateLocationNowcastDataset, teammate_location_nowcast_collate_fn


class TeammateLocationNowcastDataModule(BaseDataModule):
    """
    Lightning DataModule for multi-agent teammate location nowcast dataset.
    
    Provides train/validation/test splits and data loading functionality 
    for multi-agent teammate location nowcast using multi-label classification.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize Teammate Location Nowcast DataModule.
        
        Args:
            cfg: Configuration dictionary containing all required parameters
        """
        super().__init__(cfg)
        
        # Multi-agent teammate location nowcast parameters
        self.num_agents = cfg.data.num_pov_agents
        
        # Store dataset info for model configuration
        self.num_places = None
        self.place_names = None
        self.place_to_idx = None
    
    def _create_base_dataset(self):
        """Create the base teammate location nowcast dataset."""
        return TeammateLocationNowcastDataset(cfg=self.cfg)
    
    def _get_collate_fn(self):
        """Get the collate function for teammate location nowcast."""
        return teammate_location_nowcast_collate_fn
    
    def _copy_dataset_attributes(self, base_dataset, partition_dataset):
        """Copy dataset attributes for place-based classification."""
        for attr in ['place_names', 'place_to_idx', 'idx_to_place', 'num_places']:
            if hasattr(base_dataset, attr):
                setattr(partition_dataset, attr, getattr(base_dataset, attr))
    
    def _print_partition_info(self, df, partition_name: str):
        """Print partition information."""
        total_samples = len(df)
        logger.info(f"{partition_name} partition: {total_samples} samples")
        
        # Print team distribution
        if 'pov_team_side' in df.columns:
            team_counts = df['pov_team_side'].value_counts()
            logger.info(f"  Team distribution: {dict(team_counts)}")
        
        if hasattr(self, 'place_names') and self.place_names:
            logger.info(f"  Number of places: {len(self.place_names)}")
    
    def _store_dataset_info(self, base_dataset):
        """Store dataset information in config."""
        self.num_places = base_dataset.num_places
        self.place_names = base_dataset.place_names
        self.place_to_idx = base_dataset.place_to_idx
        
        # Update config with place information
        self.cfg.num_places = self.num_places
        self.cfg.place_names = self.place_names
        
        logger.info(f"Full dataset: {len(base_dataset)} samples")
        logger.info(f"Number of agents: {self.num_agents}")
        logger.info("Task: Teammate location nowcast (current position prediction)")
        logger.info(f"Number of unique places: {self.num_places}")
        logger.info(f"Places: {self.place_names[:10]}{'...' if len(self.place_names) > 10 else ''}")
