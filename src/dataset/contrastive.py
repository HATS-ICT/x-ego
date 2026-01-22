"""
Contrastive Learning Dataset for Team Alignment (Stage 1)

This dataset provides multi-agent video data for contrastive learning,
where positive pairs are agents from the same team at the same time,
and negative pairs are agents from different teams/times.

Key features:
- Loads from contrastive.csv (no labels, just metadata and teammate info)
- Allows variable number of agents (supports dead teammates)
- Creates alignment matrix where positive pairs are teammate squares
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from .dataset_utils import (
    init_video_processor,
    construct_video_path,
    load_video_clip,
    transform_video,
)


def plot_video_example(
    video_clip: torch.Tensor,
    save_path: Union[str, Path],
) -> None:
    """
    Plot a video clip as a 4x5 grid of frames and save to file.

    Args:
        video_clip: Tensor of shape (20, 3, 306, 306) [T, C, H, W].
        save_path: Path to save the figure (e.g. .png). Will be saved in artifact/ folder.
    """
    save_path = Path(save_path)
    artifact_dir = Path("artifact")
    artifact_dir.mkdir(exist_ok=True)
    full_save_path = artifact_dir / save_path.name
    
    n_frames = video_clip.shape[0]
    if n_frames != 20:
        logger.warning(f"plot_video_example expects 20 frames, got {n_frames}; grid may be incomplete")

    # [T, C, H, W] -> numpy, then [T, H, W, C] for imshow
    arr = video_clip.detach().float().numpy()
    if arr.max() > 1.0:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    arr = np.transpose(arr, (0, 2, 3, 1))  # [T, H, W, C]

    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        if i < n_frames:
            ax.imshow(arr[i])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(full_save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved video example to {full_save_path}")


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning (team alignment).
    
    Loads from contrastive.csv which contains:
    - Metadata: match_id, round_num, partition, timestamps, etc.
    - Teammate info: num_alive_teammates, teammate_0_id through teammate_4_id
      (alive agents are guaranteed to be the first k teammate_idx where k = num_alive_teammates)
    
    No labels are used - contrastive alignment is based on batch structure.
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize Contrastive Learning Dataset.
        
        Args:
            cfg: Configuration dictionary containing dataset parameters
        """
        self.cfg = cfg
        
        # Load label CSV file
        self.df = pl.read_csv(self.cfg.data.label_path, null_values=[])
        
        # Partition filtering
        if self.cfg.data.partition != 'all':
            initial_count = len(self.df)
            self.df = self.df.filter(pl.col('partition') == self.cfg.data.partition)
            filtered_count = len(self.df)
            logger.info(f"Filtered dataset from {initial_count} to {filtered_count} samples for partition '{self.cfg.data.partition}'")
        
        # Initialize video processor
        self._init_video_processor()
        
        logger.info(f"Contrastive dataset initialized with {len(self.df)} samples")
    
    def _init_video_processor(self):
        """Initialize video processor based on model type from config."""
        self.video_processor, self.processor_type = init_video_processor(self.cfg)
        self.model_type = self.cfg.model.encoder.video.model_type
    
    def _get_alive_agent_ids(self, row: Dict[str, Any]) -> List[str]:
        num_alive = int(row['num_alive_teammates'])
        agent_ids = []
        for i in range(num_alive):
            agent_id = row[f'teammate_{i}_id']
            agent_ids.append(str(agent_id))
        return agent_ids
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample for contrastive learning.
        
        Returns:
            dict: Dictionary containing:
                - 'video': Multi-agent video features [A, ...] where A is variable
                - 'num_agents': Number of valid agents
                - 'pov_team_side': Team side (string)
                - 'pov_team_side_encoded': Team side encoded as int
                - 'agent_ids': List of agent IDs
        """
        row = self.df.row(idx, named=True)
        
        # Extract metadata
        start_seconds = row['start_seconds']
        end_seconds = row['end_seconds']
        match_id = row['match_id']
        round_num = row['round_num']
        pov_team_side = str(row['pov_team_side']).upper()
        original_csv_idx = row['idx']
        agent_ids = self._get_alive_agent_ids(row)
        
        agent_videos = []
        for agent_id in agent_ids:
            video_path = construct_video_path(self.cfg, match_id, agent_id, round_num)
            video_clip = load_video_clip(self.cfg, video_path, start_seconds, end_seconds) # [T, C, H, W]
            
            ## Temporary: Debug, plot each video clip and save png
            plot_video_example(video_clip, f"debug_video_{agent_id}.png")
            
            video_features = transform_video(self.video_processor, self.processor_type, video_clip)
            agent_videos.append(video_features)
        
        multi_agent_video = torch.stack(agent_videos, dim=0)  # [A, T, C, H, W]
        
        # Encode team side
        pov_team_side_encoded = 1 if pov_team_side == 'CT' else 0
        
        return {
            'video': multi_agent_video,
            'num_agents': len(agent_ids),
            'pov_team_side': pov_team_side,
            'pov_team_side_encoded': torch.tensor(pov_team_side_encoded, dtype=torch.long),
            'agent_ids': agent_ids,
            'original_csv_idx': original_csv_idx,
        }