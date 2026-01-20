"""
Contrastive Learning Dataset for Team Alignment (Stage 1)

This dataset provides multi-agent video data for contrastive learning,
where positive pairs are agents from the same team at the same time,
and negative pairs are agents from different teams/times.

Key difference from previous datasets:
- Allows variable number of agents (supports dead teammates)
- Only needs video embeddings and team information (no task labels)
- Creates alignment matrix where positive pairs are teammate squares
"""

import logging
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .base import BaseVideoDataset
except ImportError:
    from base import BaseVideoDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def contrastive_collate_fn(batch):
    """
    Custom collate function for contrastive learning dataset.
    
    Handles variable number of agents per sample by padding to max agents in batch.
    
    Args:
        batch: List of dictionaries containing:
            - 'video': Video features tensor [A, ...] where A is variable
            - 'num_agents': Number of valid agents in this sample
            - 'pov_team_side': String indicating team side
            - 'pov_team_side_encoded': Team side encoded as int
            - 'agent_ids': List of agent IDs
            - 'agent_mask': Boolean mask indicating valid agents
    
    Returns:
        Dictionary with batched tensors, including agent_mask for variable agents
    """
    collated = {}
    
    # Find max agents in batch
    max_agents = max(item['video'].shape[0] for item in batch)
    
    # Determine if using precomputed embeddings (3D: [A, embed_dim]) or raw video (5D: [A, T, C, H, W])
    sample_video = batch[0]['video']
    is_embedding = len(sample_video.shape) == 2
    
    # Pad videos to max_agents
    padded_videos = []
    agent_masks = []
    
    for item in batch:
        video = item['video']
        num_agents = video.shape[0]
        
        if num_agents < max_agents:
            # Pad with zeros
            if is_embedding:
                # [A, embed_dim] -> [max_agents, embed_dim]
                pad_shape = (max_agents - num_agents, video.shape[1])
            else:
                # [A, T, C, H, W] -> [max_agents, T, C, H, W]
                pad_shape = (max_agents - num_agents,) + video.shape[1:]
            
            padding = torch.zeros(pad_shape, dtype=video.dtype)
            video = torch.cat([video, padding], dim=0)
        
        padded_videos.append(video)
        
        # Create agent mask: True for valid agents, False for padding
        mask = torch.zeros(max_agents, dtype=torch.bool)
        mask[:num_agents] = True
        agent_masks.append(mask)
    
    collated['video'] = torch.stack(padded_videos, dim=0)  # [B, max_A, ...]
    collated['agent_mask'] = torch.stack(agent_masks, dim=0)  # [B, max_A]
    collated['num_agents'] = torch.tensor([item['num_agents'] for item in batch])  # [B]
    
    # Handle other keys
    for key in batch[0].keys():
        if key in ['video', 'num_agents']:
            continue
        
        values = [item[key] for item in batch]
        
        if key in ['pov_team_side', 'agent_ids']:
            # Keep string/list values as lists
            collated[key] = values
        else:
            # For tensors, use default collate
            collated[key] = torch.utils.data.default_collate(values)
    
    return collated


class ContrastiveDataset(BaseVideoDataset, Dataset):
    """
    Dataset for contrastive learning (team alignment).
    
    Provides multi-agent video data where:
    - Each sample contains videos from 1-5 teammates at the same timestamp
    - Supports variable number of agents (dead teammates excluded)
    - Only provides video embeddings and team info (no task-specific labels)
    
    The contrastive learning objective aligns agents from the same team
    while separating agents from different batches (different teams/times).
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize Contrastive Learning Dataset.
        
        Args:
            cfg: Configuration dictionary containing dataset parameters
        """
        # Temporarily override contrastive setting to get full 5 agents
        # (base class uses this to determine num_agents_to_sample)
        original_contrastive = cfg.model.contrastive.enable
        cfg.model.contrastive.enable = True
        
        # Initialize base class
        super().__init__(cfg)
        
        # Restore original setting
        cfg.model.contrastive.enable = original_contrastive
        
        # Contrastive-specific configuration
        self.allow_variable_agents = cfg.data.allow_variable_agents
        self.min_agents = cfg.data.min_agents
        
        # Override num_agents_to_sample: always try to get all 5
        self.num_agents_to_sample = 5
        
        logger.info(f"Contrastive dataset initialized with {len(self.df)} samples")
        logger.info(f"Allow variable agents: {self.allow_variable_agents}")
        logger.info(f"Minimum agents: {self.min_agents}")
    
    def _init_label_creator(self):
        """Override: No label creator needed for contrastive learning."""
        self.label_creator = None
    
    def _extract_unique_places(self) -> List[str]:
        """Extract unique place names (required by base class but not used for contrastive)."""
        places = set()
        
        # Extract places from all teammate location columns
        for i in range(5):
            place_col = f'teammate_{i}_place'
            if place_col in self.df.columns:
                places.update(self.df[place_col].unique())
        
        # Remove empty/null values
        places = {p for p in places if p and str(p).strip() and str(p) != 'nan'}
        
        return sorted(list(places))
    
    def _get_team_player_data(self, row: pd.Series, player_idx: int) -> Dict:
        """Extract team player data from a row."""
        player_id = row[f'teammate_{player_idx}_id']
        
        # Check if player is valid (not dead/missing)
        # Dead players typically have empty ID or health == 0
        if not player_id or str(player_id).strip() == '' or str(player_id) == 'nan':
            return None
        
        # Check health if available
        health_col = f'teammate_{player_idx}_health'
        if health_col in row and row[health_col] == 0:
            return None
        
        return {
            'id': player_id,
            'name': row.get(f'teammate_{player_idx}_name', ''),
            'side': row.get(f'teammate_{player_idx}_side', ''),
            'X_norm': row.get(f'teammate_{player_idx}_X', 0),
            'Y_norm': row.get(f'teammate_{player_idx}_Y', 0),
            'Z_norm': row.get(f'teammate_{player_idx}_Z', 0),
            'place': row.get(f'teammate_{player_idx}_place', ''),
        }
    
    def _get_alive_team_players(self, row: pd.Series) -> List[Dict]:
        """Get all alive teammates from the team."""
        team_players = []
        
        for i in range(5):
            try:
                player_data = self._get_team_player_data(row, i)
                if player_data is not None:
                    player_data['agent_position'] = i  # Store original position for embedding lookup
                    team_players.append(player_data)
            except KeyError:
                continue
        
        return team_players
    
    def _select_agents_variable(self, team_players: List[Dict]) -> List[Dict]:
        """
        Select agents, allowing variable counts.
        
        Unlike the base class, this doesn't pad to a fixed number.
        Returns as many alive agents as available (up to 5).
        """
        if len(team_players) < self.min_agents:
            return None  # Will be filtered out
        
        return team_players[:5]  # Take up to 5 agents
    
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
        row = self.df.iloc[idx]
        start_seconds = row['normalized_start_seconds']
        end_seconds = row['normalized_end_seconds']
        match_id = row['match_id']
        round_num = row['round_num']
        pov_team_side = row['pov_team_side'].upper()
        
        # Get original CSV index for pre-computed embeddings
        original_csv_idx = row.get('original_csv_idx', idx)
        
        # Get alive team players
        team_players = self._get_alive_team_players(row)
        
        # Select agents (variable count)
        selected_agents = self._select_agents_variable(team_players)
        
        if selected_agents is None:
            # Not enough agents - this should be filtered during setup
            # For safety, return a minimal sample that will be masked
            return self._create_minimal_sample(pov_team_side)
        
        # Load videos or embeddings
        agent_videos = []
        agent_ids = []
        
        if self.use_precomputed_embeddings:
            for agent in selected_agents:
                agent_position = agent['agent_position']
                embedding = self._load_embedding(original_csv_idx, agent_position)
                agent_videos.append(embedding)
                agent_ids.append(agent['id'])
            
            multi_agent_video = torch.stack(agent_videos, dim=0)  # [A, embed_dim]
        else:
            for agent in selected_agents:
                video_path = self._construct_video_path(match_id, agent['id'], round_num)
                video_clip = self._load_video_clip_with_decord(video_path, start_seconds, end_seconds)
                video_features = self._transform_video(video_clip)
                agent_videos.append(video_features)
                agent_ids.append(agent['id'])
            
            multi_agent_video = torch.stack(agent_videos, dim=0)  # [A, T, C, H, W]
        
        # Encode team side
        pov_team_side_encoded = 1 if pov_team_side == 'CT' else 0
        
        return {
            'video': multi_agent_video,
            'num_agents': len(selected_agents),
            'pov_team_side': pov_team_side,
            'pov_team_side_encoded': torch.tensor(pov_team_side_encoded, dtype=torch.long),
            'agent_ids': agent_ids,
            'original_csv_idx': original_csv_idx,
        }
    
    def _create_minimal_sample(self, pov_team_side: str) -> Dict:
        """Create a minimal sample for edge cases (not enough agents)."""
        pov_team_side_encoded = 1 if pov_team_side == 'CT' else 0
        
        if self.use_precomputed_embeddings:
            # Get embedding dimension from config
            from models.video_encoder import get_embed_dim_for_model_type
            embed_dim = get_embed_dim_for_model_type(self.cfg.model.encoder.video.model_type)
            video = torch.zeros(1, embed_dim)
        else:
            # Create placeholder video
            num_frames = int(self.fixed_duration_seconds * self.target_fps)
            video = torch.zeros(1, num_frames, 3, 224, 224)
        
        return {
            'video': video,
            'num_agents': 0,  # Will be masked
            'pov_team_side': pov_team_side,
            'pov_team_side_encoded': torch.tensor(pov_team_side_encoded, dtype=torch.long),
            'agent_ids': [],
            'original_csv_idx': -1,
        }


if __name__ == "__main__":
    print("ContrastiveDataset test placeholder")
