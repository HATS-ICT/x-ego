"""
Contrastive Learning DataModule (Stage 1)

Lightning DataModule for team alignment contrastive learning.
Handles variable number of agents per sample.
"""

from typing import Dict, Any, Iterator
from rich import print as rprint
import torch
from torch.utils.data import DataLoader, Sampler

from .base import BaseDataModule
from ..dataset.contrastive import ContrastiveDataset
from ..dataset.collate import contrastive_collate_fn


class FixedVideoBatchSampler(Sampler[list[tuple[int, int]]]):
    """
    Batch sampler that makes contrastive batch size mean total videos.

    Each yielded element is a list of (dataset_index, agents_to_load). If a
    full team would exceed the video budget, only the first agents that fit are
    loaded and the rest of that team is dropped.
    """

    def __init__(
        self,
        agent_counts: list[int],
        video_batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        if video_batch_size <= 0:
            raise ValueError(f"video_batch_size must be positive, got {video_batch_size}")

        self.agent_counts = [int(count) for count in agent_counts]
        self.video_batch_size = int(video_batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._epoch = 0

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        indices = self._ordered_indices()
        batch = []
        remaining = self.video_batch_size

        for idx in indices:
            agent_count = min(self.agent_counts[idx], self.video_batch_size)
            if agent_count <= 0:
                continue

            if agent_count <= remaining:
                batch.append((idx, agent_count))
                remaining -= agent_count
                if remaining == 0:
                    yield batch
                    batch = []
                    remaining = self.video_batch_size
                continue

            batch.append((idx, remaining))
            yield batch
            batch = []
            remaining = self.video_batch_size

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        full_batches = 0
        has_partial_batch = False
        remaining = self.video_batch_size

        for count in self.agent_counts:
            agent_count = min(count, self.video_batch_size)
            if agent_count <= 0:
                continue

            if agent_count <= remaining:
                remaining -= agent_count
                has_partial_batch = True
                if remaining == 0:
                    full_batches += 1
                    has_partial_batch = False
                    remaining = self.video_batch_size
                continue

            full_batches += 1
            has_partial_batch = False
            remaining = self.video_batch_size

        if has_partial_batch and not self.drop_last:
            return full_batches + 1
        return full_batches

    def _ordered_indices(self) -> list[int]:
        if not self.shuffle:
            return list(range(len(self.agent_counts)))

        generator = torch.Generator()
        generator.manual_seed(torch.initial_seed() + self._epoch)
        self._epoch += 1
        return torch.randperm(len(self.agent_counts), generator=generator).tolist()


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

    def _create_video_dataloader(
        self,
        dataset,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        data_cfg = self.cfg.data
        batch_sampler = FixedVideoBatchSampler(
            agent_counts=dataset.df['num_alive_teammates'].to_list(),
            video_batch_size=data_cfg.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_mem,
            persistent_workers=data_cfg.persistent_workers if data_cfg.num_workers > 0 else False,
            prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
            collate_fn=self._get_collate_fn(),
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader with fixed total-video batches."""
        return self._create_video_dataloader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Return validation dataloader with fixed total-video batches."""
        if self.val_dataset is None:
            return None

        return self._create_video_dataloader(
            self.val_dataset,
            shuffle=False,
            drop_last=self._get_val_drop_last(),
        )

    def test_dataloader(self) -> DataLoader | None:
        """Return test dataloader with fixed total-video batches."""
        if self.test_dataset is None:
            return None

        return self._create_video_dataloader(
            self.test_dataset,
            shuffle=False,
            drop_last=self._get_test_drop_last(),
        )
    
    def _copy_dataset_attributes(self, base_dataset, partition_dataset):
        """Copy dataset attributes."""
        attrs = ['video_processor', 'processor_type', 'model_type', 'time_offsets']
        for attr in attrs:
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
        rprint(f"[green]OK[/green] Full contrastive dataset: [bold]{len(base_dataset):,}[/bold] samples")
    
    def _get_val_drop_last(self) -> bool:
        """Drop last batch in validation for consistent batch sizes."""
        return True
    
    def _get_test_drop_last(self) -> bool:
        """Drop last batch in test for consistent batch sizes."""
        return True
