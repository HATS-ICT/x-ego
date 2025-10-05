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
    from ..dataset.teammate_opponent_traj_prediction import TeammateOpponentTrajPredictionDataset, trajectory_prediction_collate_fn
except ImportError:
    from dataset.teammate_opponent_traj_prediction import TeammateOpponentTrajPredictionDataset, trajectory_prediction_collate_fn


class TeammateOpponentTrajPredictionDataModule(BaseDataModule):
    """
    Lightning DataModule for trajectory prediction dataset.
    
    Provides train/validation/test splits and data loading functionality 
    for trajectory prediction tasks with flexible POV and target team selection.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize Trajectory Prediction DataModule.
        
        Args:
            cfg: Configuration dictionary containing all required parameters
        """
        super().__init__(cfg)
        
        # Trajectory prediction parameters
        self.num_agents = cfg.data.num_pov_agents
        self.video_length_sec = cfg.data.video_length_sec
        self.total_trajectory_sec = cfg.data.total_trajectory_sec
        self.trajectory_sample_rate = cfg.data.trajectory_sample_rate
    
    def _create_base_dataset(self):
        """Create the base trajectory prediction dataset."""
        return TeammateOpponentTrajPredictionDataset(cfg=self.cfg)
    
    def _get_collate_fn(self):
        """Get the collate function for trajectory prediction."""
        return trajectory_prediction_collate_fn
    
    def _validate_additional_paths(self):
        """Validate H5 file exists."""
        from pathlib import Path
        h5_filename = self.cfg.data.labels_filename.replace('.csv', '.h5')
        h5_path = Path(self.cfg.path.data) / self.cfg.data.labels_folder / h5_filename
        
        if not h5_path.exists():
            logger.error(f"H5 trajectory file not found: {h5_path}")
            raise FileNotFoundError(f"H5 trajectory file not found: {h5_path}")
    
    def _print_partition_info(self, df, partition_name: str):
        """Print partition information."""
        total_samples = len(df)
        logger.info(f"{partition_name} partition: {total_samples} samples")
        logger.info(f"  Video length: {self.video_length_sec}s")
        logger.info(f"  Trajectory length: {self.total_trajectory_sec}s at {self.trajectory_sample_rate}Hz")
    
    def _store_dataset_info(self, base_dataset):
        """Store dataset information in config."""
        logger.info(f"Full dataset: {len(base_dataset)} samples")
        logger.info(f"Number of POV agents: {self.num_agents}")
        logger.info(f"Video length: {self.video_length_sec}s")
        logger.info(f"Trajectory prediction: {self.total_trajectory_sec}s at {self.trajectory_sample_rate}Hz")
        logger.info(f"Number of trajectory timepoints: {int(self.total_trajectory_sec * self.trajectory_sample_rate)}")
        logger.info(f"POV-Target sampling: random (all 4 combinations possible)")
        
        # Store in config for model
        self.cfg.num_trajectory_timepoints = int(self.total_trajectory_sec * self.trajectory_sample_rate)
        self.cfg.trajectory_output_dim = 2  # X, Y coordinates
        self.cfg.num_target_players = 5  # Always predicting 5 players