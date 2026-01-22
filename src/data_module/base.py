import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from torch.utils.data import DataLoader
import lightning as L
import polars as pl
from rich import print as rprint


class BaseDataModule(L.LightningDataModule, ABC):
    """
    Base Lightning DataModule providing common functionality for all X-EGO data modules.
    
    This class handles common patterns like:
    - Configuration management
    - Path building and validation
    - Dataset partitioning
    - DataLoader creation with consistent parameters
    - Logging and error handling
    """
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize Base DataModule.
        
        Args:
            cfg: Configuration dictionary containing all required parameters
        """
        super().__init__()
        
        # Store config
        self.cfg = cfg
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # State tracking for resumable training (optional)
        self._train_state = None
        self._val_state = None
    
    def _build_label_path(self) -> Path:
        """Build label path from config components."""
        data_cfg = self.cfg.data
        return Path(self.cfg.path.data) / data_cfg.labels_folder / data_cfg.labels_filename
    
    def _build_data_root_path(self) -> Path:
        """Build data root path from config."""
        return Path(self.cfg.path.data)
    
    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        # Check label path if it exists in config
        data_cfg = self.cfg.data
        if hasattr(data_cfg, 'labels_filename'):
            label_path = self._build_label_path()
            data_cfg.label_path = label_path
            
            if not label_path.exists():
                rprint(f"[red]✗[/red] Label CSV file not found: [bold]{label_path}[/bold]")
                raise FileNotFoundError(f"Label CSV file not found: {label_path}")
        
        # Check data root
        data_root = self._build_data_root_path()
        if not data_root.exists():
            rprint(f"[red]✗[/red] Data directory not found: [bold]{data_root}[/bold]")
            raise FileNotFoundError(f"Data directory not found: {data_root}")
        
        # Additional path validation can be overridden by subclasses
        self._validate_additional_paths()
    
    def _validate_additional_paths(self) -> None:
        """Override this method to add additional path validation."""
        pass
    
    def _create_partition_dataset(self, base_dataset, partition: str):
        """
        Create a dataset for a specific partition.
        
        Args:
            base_dataset: The base dataset to partition
            partition: Partition name ('train', 'val', 'test')
            
        Returns:
            Partitioned dataset
        """
        partition_dataset = copy.copy(base_dataset)
        # Use Polars filter method instead of boolean indexing
        partition_df = base_dataset.df.filter(pl.col('partition') == partition)
        partition_dataset.df = partition_df
        
        # Copy additional attributes if needed
        self._copy_dataset_attributes(base_dataset, partition_dataset)
        
        # Print partition info
        self._print_partition_info(partition_df, partition)
        
        return partition_dataset
    
    def _copy_dataset_attributes(self, base_dataset, partition_dataset) -> None:
        """Override this method to copy additional dataset attributes."""
        pass
    
    def _print_partition_info(self, df, partition_name: str) -> None:
        """Print partition information. Override for custom info."""
        total_samples = len(df)
        rprint(f"[cyan]{partition_name}[/cyan] partition: [bold]{total_samples:,}[/bold] samples")
    
    def _create_dataloader(
        self, 
        dataset, 
        shuffle: bool = False, 
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None
    ) -> DataLoader:
        """
        Create a DataLoader with consistent parameters.
        
        Args:
            dataset: Dataset to create loader for
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            collate_fn: Collate function for batching
            
        Returns:
            Configured DataLoader
        """
        data_cfg = self.cfg.data
        return DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            shuffle=shuffle,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_mem,
            persistent_workers=data_cfg.persistent_workers if data_cfg.num_workers > 0 else False,
            prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
            collate_fn=collate_fn,
            drop_last=drop_last
        )
    
    @abstractmethod
    def _create_base_dataset(self):
        """Create the base dataset. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_collate_fn(self):
        """Get the collate function for this dataset type. Must be implemented by subclasses."""
        pass
    
    def prepare_data(self) -> None:
        """Check if required data files exist."""
        self._validate_paths()
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            rprint(f"[blue]->[/blue] Loading [bold]{self.__class__.__name__}[/bold] dataset and initializing processors...")
            base_dataset = self._create_base_dataset()
            
            # Store dataset info for model configuration
            self._store_dataset_info(base_dataset)
            
            # Create partition datasets
            self.train_dataset = self._create_partition_dataset(base_dataset, 'train')
            self.val_dataset = self._create_partition_dataset(base_dataset, 'val')
            
            # Apply any previously loaded states
            self._apply_saved_states()
            
            rprint(f"[green]CHECK[/green] Dataset split - [cyan]Train[/cyan]: [bold]{len(self.train_dataset):,}[/bold], [cyan]Val[/cyan]: [bold]{len(self.val_dataset):,}[/bold]")
        
        elif stage == "test":
            rprint(f"[blue]->[/blue] Loading [bold]{self.__class__.__name__}[/bold] dataset for testing...")
            base_dataset = self._create_base_dataset()
            
            # Store dataset info for model configuration
            self._store_dataset_info(base_dataset)
            
            try:
                self.test_dataset = self._create_partition_dataset(base_dataset, 'test')
                rprint(f"[green]CHECK[/green] Test dataset size: [bold]{len(self.test_dataset):,}[/bold]")
            except Exception as e:
                rprint(f"[yellow]WARN[/yellow] No test split found: [dim]{e}[/dim]")
                self.test_dataset = None
    
    def _store_dataset_info(self, base_dataset) -> None:
        """Store dataset information in config. Override for custom info."""
        pass
    
    def _apply_saved_states(self) -> None:
        """Apply any saved states for resumable training."""
        if hasattr(self, '_train_state') and self._train_state is not None and self.train_dataset is not None:
            if hasattr(self.train_dataset, 'load_state_dict'):
                self.train_dataset.load_state_dict(self._train_state)
            self._train_state = None
            
        if hasattr(self, '_val_state') and self._val_state is not None and self.val_dataset is not None:
            if hasattr(self.val_dataset, 'load_state_dict'):
                self.val_dataset.load_state_dict(self._val_state)
            self._val_state = None
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return self._create_dataloader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self._get_collate_fn()
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Return validation dataloader."""
        if self.val_dataset is None:
            return None
        
        return self._create_dataloader(
            self.val_dataset,
            shuffle=False,
            drop_last=self._get_val_drop_last(),
            collate_fn=self._get_collate_fn()
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Return test dataloader if test dataset exists."""
        if self.test_dataset is None:
            return None
        
        return self._create_dataloader(
            self.test_dataset,
            shuffle=False,
            drop_last=self._get_test_drop_last(),
            collate_fn=self._get_collate_fn()
        )
    
    def _get_val_drop_last(self) -> bool:
        """Whether to drop last batch in validation. Override if needed."""
        return False
    
    def _get_test_drop_last(self) -> bool:
        """Whether to drop last batch in testing. Override if needed."""
        return False
    
    # Optional state management methods
    def state_dict(self) -> Dict[str, Any]:
        """Return the state dictionary for resumable training."""
        state = {}
        
        if self.train_dataset is not None and hasattr(self.train_dataset, 'state_dict'):
            state['train_dataset'] = self.train_dataset.state_dict()
        
        if self.val_dataset is not None and hasattr(self.val_dataset, 'state_dict'):
            state['val_dataset'] = self.val_dataset.state_dict()
            
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state dictionary for resumable training."""
        self._train_state = state_dict.get('train_dataset')
        self._val_state = state_dict.get('val_dataset')
        
        # Apply states if datasets are already created
        self._apply_saved_states()
        
        train_status = "[green]train[/green]" if self._train_state else "[dim]no train[/dim]"
        val_status = "[green]val[/green]" if self._val_state else "[dim]no val[/dim]"
        rprint(f"[blue]->[/blue] Loaded DataModule state for {train_status} and {val_status}")
    
    # Optional epoch hooks
    def on_train_epoch_start(self) -> None:
        """Called at the start of each training epoch."""
        if self.train_dataset is not None and hasattr(self.train_dataset, 'on_epoch_start'):
            self.train_dataset.on_epoch_start()
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        if self.train_dataset is not None and hasattr(self.train_dataset, 'on_epoch_end'):
            self.train_dataset.on_epoch_end()
    
    def on_validation_epoch_start(self) -> None:
        """Called at the start of each validation epoch."""
        if self.val_dataset is not None and hasattr(self.val_dataset, 'on_epoch_start'):
            self.val_dataset.on_epoch_start()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        if self.val_dataset is not None and hasattr(self.val_dataset, 'on_epoch_end'):
            self.val_dataset.on_epoch_end()
