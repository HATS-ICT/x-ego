import os
from pathlib import Path
from typing import Optional, Dict, Any
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
    from ..dataset import CTFMContrastiveDataset, video_audio_text_collate_fn
except ImportError:
    from dataset import CTFMContrastiveDataset, video_audio_text_collate_fn


class CTFMContrastiveDataModule(BaseDataModule):
    """
    Lightning DataModule for CTFM contrastive dataset handling video, audio, and text data.
    
    Provides train/validation splits and data loading functionality for contrastive learning.
    No labels are used - only video paths for self-supervised learning.
    
    Implements state_dict and load_state_dict for resumable training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CTFM Contrastive DataModule.
        
        Args:
            config: Configuration dictionary containing all required parameters
        """
        super().__init__(config)
    
    def _validate_additional_paths(self) -> None:
        """Validate contrastive-specific paths."""
        data_config = self.config['data']
        
        # Build data path using consistent path configuration  
        data_path_csv_path = str(Path(self.config['path']['data']) / data_config['data_path_csv_filename'])
        
        # Check if data files exist
        if not os.path.exists(data_path_csv_path):
            logger.error(f"CSV file not found: {data_path_csv_path}")
            raise FileNotFoundError(f"CSV file not found: {data_path_csv_path}")
        
        # Check if data directories exist
        data_root = self._build_data_root_path()
        
        youtube_dir = data_root / data_config['youtube_folder']
        recording_dir = data_root / data_config['recording_folder']
        
        if not youtube_dir.exists():
            logger.error(f"Youtube directory not found: {youtube_dir}")
            raise FileNotFoundError(f"Youtube directory not found: {youtube_dir}")
        
        if not recording_dir.exists():
            logger.error(f"Recording directory not found: {recording_dir}")
            raise FileNotFoundError(f"Recording directory not found: {recording_dir}")
    
    def _create_base_dataset(self):
        """Create the base contrastive dataset."""
        return CTFMContrastiveDataset(config=self.config)
    
    def _get_collate_fn(self):
        """Get the collate function for contrastive learning."""
        return video_audio_text_collate_fn
    
    def _get_val_drop_last(self) -> bool:
        """Drop last batch for validation to prevent top-k issues."""
        return True
    
    def _get_test_drop_last(self) -> bool:
        """Drop last batch for testing to prevent top-k issues."""
        return True
    
    def _create_partition_dataset(self, base_dataset, partition: str):
        """Create a dataset for a specific partition using filter_by_split."""
        # For contrastive dataset, we need to create a new instance and filter
        partition_dataset = CTFMContrastiveDataset(config=self.config)
        try:
            partition_dataset.filter_by_split(partition)
        except ValueError as e:
            if partition == 'test':
                logger.warning(f"No test split found: {e}")
                return None
            else:
                raise e
        
        self._print_partition_info_contrastive(partition_dataset, partition)
        return partition_dataset
    
    def _print_partition_info_contrastive(self, dataset, partition_name: str):
        """Print partition information for contrastive dataset."""
        if dataset is not None:
            total_samples = len(dataset)
            logger.info(f"{partition_name} partition: {total_samples} samples")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage using pre-created splits from CSV."""
        if stage == "fit" or stage is None:
            logger.info("Loading contrastive dataset and initializing processors...")
            
            # Create partition datasets using the custom method
            self.train_dataset = self._create_partition_dataset(None, 'train')
            self.val_dataset = self._create_partition_dataset(None, 'val')
            
            # Apply any previously loaded states
            self._apply_saved_states()
            
            logger.info(f"Dataset split - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        
        elif stage == "test":
            logger.info("Loading contrastive dataset for testing...")
            self.test_dataset = self._create_partition_dataset(None, 'test')
            if self.test_dataset:
                logger.info(f"Test dataset size: {len(self.test_dataset)}")
            else:
                logger.warning("No test split found")
        


# Example usage and testing
if __name__ == "__main__":
    try:
        print("Creating CTFM DataModule...")
        
        # Load config from file for testing
        import yaml
        with open("configs/ctfm_contrastive.yaml", 'r') as f:
            full_config = yaml.safe_load(f)
            
        # Override some settings for testing
        full_config['data']['batch_size'] = 2
        full_config['data']['num_workers'] = 0
        full_config['data']['persistent_workers'] = False
        full_config['data']['pin_mem'] = False
        
        # Create datamodule
        datamodule = CTFMContrastiveDataModule(config=full_config)
        
        print("Preparing data...")
        datamodule.prepare_data()
        
        print("Setting up data...")
        datamodule.setup("fit")
        
        print(f"Train dataset size: {len(datamodule.train_dataset) if datamodule.train_dataset else 0}")
        print(f"Val dataset size: {len(datamodule.val_dataset) if datamodule.val_dataset else 0}")
        
        print("Testing single sample...")
        # Test loading a single sample
        train_loader = datamodule.train_dataloader()
        sample_batch = next(iter(train_loader))
        
        print(f"Batch keys: {list(sample_batch.keys())}")
        if 'video' in sample_batch:
            print(f"Video shape: {sample_batch['video'].shape}")
        if 'audio' in sample_batch:
            print(f"Audio shape: {sample_batch['audio'].shape}")
        if 'text' in sample_batch:
            print(f"Text shape: {sample_batch['text'].shape}")
        print(f"Video paths: {sample_batch['video_path']}")
        if 'raw_text' in sample_batch:
            print(f"Raw text: {sample_batch['raw_text']}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
