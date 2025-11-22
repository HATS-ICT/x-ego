import logging
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

try:
    from .base import BaseVideoDataset
except ImportError:
    from base import BaseVideoDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def trajectory_prediction_collate_fn(batch):
    """
    Custom collate function for trajectory prediction dataset.
    
    Args:
        batch: List of dictionaries containing:
            - 'video': Video features tensor [A, T, C, H, W] where A is num_agents
            - 'trajectories': Trajectory tensor [N, 60, 2] where N is num_target_players
            - 'pov_team_side': String indicating POV team side
            - 'target_team_side': String indicating target team side
            - 'agent_ids': List of agent IDs used for video
            - 'target_player_ids': List of player IDs for trajectories
    
    Returns:
        Dictionary with batched tensors and lists of string values
    """
    collated = {}
    
    # Handle each key appropriately
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key in ['pov_team_side', 'target_team_side', 'agent_ids', 'target_player_ids']:
            # Keep string/list values as lists
            collated[key] = values
        else:
            # For tensors (video, trajectories), use default collate
            collated[key] = torch.utils.data.default_collate(values)
    
    # Encode team sides
    team_encoding = {'t': 0, 'ct': 1}
    pov_team_encoded = torch.tensor([team_encoding[side] for side in collated['pov_team_side']], dtype=torch.long)
    target_team_encoded = torch.tensor([team_encoding[side] for side in collated['target_team_side']], dtype=torch.long)
    
    collated['pov_team_side_encoded'] = pov_team_encoded
    collated['target_team_side_encoded'] = target_team_encoded
    
    return collated


class TeammateOpponentTrajPredictionDataset(BaseVideoDataset, Dataset):
    """
    Trajectory prediction dataset for all 10 players.
    
    Given videos from a subset of players from one team (pov_team_side),
    predict the trajectories of 5 players from either the same or different team (target_team_side).
    
    This allows for:
    - Self-prediction: CT videos -> CT trajectories, T videos -> T trajectories
    - Opponent-prediction: CT videos -> T trajectories, T videos -> CT trajectories
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize Trajectory Prediction Dataset.
        
        Args:
            cfg: Configuration dictionary containing dataset parameters
        """
        # Initialize base class
        super().__init__(cfg)
        
        # Trajectory-specific parameters
        self.video_length_sec = cfg.data.video_length_sec  # 5 seconds
        self.total_trajectory_sec = cfg.data.total_trajectory_sec  # 15 seconds
        self.trajectory_sample_rate = cfg.data.trajectory_sample_rate  # 4 Hz
        self.num_timepoints = int(self.total_trajectory_sec * self.trajectory_sample_rate)  # 60
        
        # Build H5 path
        h5_filename = cfg.data.labels_filename.replace('.csv', '.h5')
        self.h5_path = Path(cfg.path.data) / cfg.data.labels_folder / h5_filename
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")
        
        # Preload all trajectories into memory
        logger.info(f"Preloading trajectory data from {self.h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            self.trajectories = torch.from_numpy(f['trajectories'][:]).float()
            logger.info(f"Loaded trajectories with shape: {self.trajectories.shape}")
            logger.info(f"Memory usage: {self.trajectories.element_size() * self.trajectories.nelement() / (1024**2):.2f} MB")
        
        logger.info(f"Dataset initialized with {len(self.df)} samples")
        logger.info(f"Number of POV agents: {self.num_agents}")
        logger.info(f"Video length: {self.video_length_sec}s, Trajectory length: {self.total_trajectory_sec}s")
        logger.info(f"Trajectory sampling: {self.trajectory_sample_rate} Hz, {self.num_timepoints} timepoints")
        logger.info(f"POV-Target sampling: random (both sides independently chosen)")
    
    def _extract_unique_places(self) -> List[str]:
        """Not used for trajectory prediction, but required by base class."""
        return []
    
    def _sample_pov_and_target_sides(self) -> tuple:
        """
        Sample POV team side and target team side randomly.
        
        Both POV and target sides are chosen independently at random,
        allowing all four combinations: CT→CT, CT→T, T→CT, T→T.
        
        Returns:
            Tuple of (pov_team_side, target_team_side)
        """
        pov_side = random.choice(['ct', 't'])
        target_side = random.choice(['ct', 't'])
        return pov_side, target_side
    
    def _get_team_players(self, row: pd.Series, side: str) -> List[Dict]:
        """
        Extract players from a specific team side.
        
        Args:
            row: DataFrame row with player data
            side: Team side ('ct' or 't')
        
        Returns:
            List of player dictionaries with id, name, side
        """
        team_players = []
        
        for i in range(10):
            player_side = row[f'player_{i}_side']
            if player_side == side:
                team_players.append({
                    'id': row[f'player_{i}_id'],
                    'name': row[f'player_{i}_name'],
                    'side': player_side,
                    'player_idx': i
                })
        
        return team_players
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
                - video: Video tensor [num_agents, num_frames, C, H, W]
                - trajectories: Trajectory tensor [5, 60, 2]
                - pov_team_side: POV team side string
                - target_team_side: Target team side string
                - agent_ids: List of agent IDs for video
                - target_player_ids: List of player IDs for trajectories
        """
        row = self.df.iloc[idx]
        
        # Get original CSV index for loading pre-computed embeddings
        original_csv_idx = row.get('original_csv_idx', idx)
        
        # Sample POV and target sides
        pov_team_side, target_team_side = self._sample_pov_and_target_sides()
        
        # Get players from each team
        pov_players = self._get_team_players(row, pov_team_side)
        target_players = self._get_team_players(row, target_team_side)
        
        # Shuffle players for randomness
        random.shuffle(pov_players)
        random.shuffle(target_players)
        
        # Select agents for POV (video)
        pov_agents = self._select_agents(pov_players)
        
        # Get metadata
        match_id = row['match_id']
        round_num = row['round_num']
        h5_traj_idx = row['h5_traj_idx']
        
        # Load videos or embeddings for POV agents
        video_clips = []
        agent_ids = []
        
        # Video timing (5 seconds from start)
        video_start_seconds = row['normalized_start_seconds']
        video_end_seconds = row['normalized_video_end_seconds']
        
        if self.use_precomputed_embeddings:
            # Load pre-computed embeddings using original CSV index
            for agent_idx, agent in enumerate(pov_agents):
                agent_ids.append(agent['id'])
                embedding = self._load_embedding(original_csv_idx, agent_idx)
                video_clips.append(embedding)
            
            # Stack embeddings: [num_agents, embed_dim]
            video = torch.stack(video_clips, dim=0)
        else:
            # Load and process videos
            for agent in pov_agents:
                agent_id = agent['id']
                agent_ids.append(agent_id)
                
                # Construct video path
                video_path = self._construct_video_path(match_id, agent_id, round_num)
                
                # Load video clip
                video_clip = self._load_video_clip_with_decord(
                    video_path,
                    video_start_seconds,
                    video_end_seconds
                )
                
                # Transform video
                video_features = self._transform_video(video_clip)
                video_clips.append(video_features)
            
            # Stack video clips: [num_agents, num_frames, C, H, W]
            video = torch.stack(video_clips, dim=0)
        
        # Get trajectories for target team
        # Load from preloaded H5 data
        full_trajectories = self.trajectories[h5_traj_idx]  # Shape: (10, 60, 2)
        
        # Extract trajectories for target team players
        target_player_indices = [p['player_idx'] for p in target_players]
        target_trajectories = full_trajectories[target_player_indices]  # Shape: (5, 60, 2)
        
        target_player_ids = [p['id'] for p in target_players]
        
        return {
            'video': video,
            'trajectories': target_trajectories,
            'pov_team_side': pov_team_side,
            'target_team_side': target_team_side,
            'agent_ids': agent_ids,
            'target_player_ids': target_player_ids,
            'original_csv_idx': original_csv_idx  # Always include for embedding extraction
        }