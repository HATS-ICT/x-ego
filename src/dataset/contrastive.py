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
from typing import Dict, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from utils.dataset_utils import apply_minimap_mask


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
    
    Returns:
        Dictionary with batched tensors, including agent_mask for variable agents
    """
    collated = {}
    
    # Find max agents in batch
    max_agents = max(item['video'].shape[0] for item in batch)
    
    # Determine if using precomputed embeddings (2D: [A, embed_dim]) or raw video (4D: [A, T, C, H, W])
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


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning (team alignment).
    
    Loads from contrastive.csv which contains:
    - Metadata: match_id, round_num, partition, timestamps, etc.
    - Teammate info: teammate_0_id through teammate_4_id with health info
    
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
        self.df = pd.read_csv(self.cfg.data.label_path, keep_default_na=False)
        
        # Partition filtering
        if self.cfg.data.partition != 'all':
            initial_count = len(self.df)
            self.df = self.df[self.df['partition'] == self.cfg.data.partition].reset_index(drop=True)
            filtered_count = len(self.df)
            logger.info(f"Filtered dataset from {initial_count} to {filtered_count} samples for partition '{self.cfg.data.partition}'")
        
        # Initialize video processor
        self._init_video_processor()
        
        # Filter out samples with too few agents if allow_variable_agents is True
        if self.cfg.data.allow_variable_agents:
            # Check num_alive_teammates column if it exists
            if 'num_alive_teammates' in self.df.columns:
                initial_count = len(self.df)
                self.df = self.df[self.df['num_alive_teammates'] >= self.cfg.data.min_agents].reset_index(drop=True)
                logger.info(f"Filtered samples with <{self.cfg.data.min_agents} agents: {initial_count} -> {len(self.df)}")
        
        logger.info(f"Contrastive dataset initialized with {len(self.df)} samples")
        logger.info(f"Allow variable agents: {self.cfg.data.allow_variable_agents}")
        logger.info(f"Minimum agents: {self.cfg.data.min_agents}")
    
    def _init_video_processor(self):
        """Initialize video processor based on model type from config."""
        from transformers import AutoImageProcessor, VivitImageProcessor, VideoMAEImageProcessor, VJEPA2VideoProcessor
        from models.modules.video_encoder import MODEL_TYPE_TO_PRETRAINED
        
        model_type = self.cfg.model.encoder.video.model_type
        self.model_type = model_type
        pretrained_model = MODEL_TYPE_TO_PRETRAINED[model_type]
        
        # Different models need different processors and have different output formats
        # processor_type: 'image' uses images= param and returns pixel_values
        #                 'video' uses videos= param and returns pixel_values_videos
        if model_type in ['siglip', 'siglip2', 'clip', 'dinov2']:
            # Image-based models: AutoImageProcessor, returns pixel_values
            self.video_processor = AutoImageProcessor.from_pretrained(pretrained_model)
            self.processor_type = 'image'
        elif model_type == 'vivit':
            # ViViT: specific image processor, returns pixel_values
            self.video_processor = VivitImageProcessor.from_pretrained(pretrained_model)
            self.processor_type = 'image'
        elif model_type == 'videomae':
            # VideoMAE: specific image processor, returns pixel_values
            self.video_processor = VideoMAEImageProcessor.from_pretrained(pretrained_model)
            self.processor_type = 'image'
        elif model_type == 'vjepa2':
            # VJEPA2: video processor, returns pixel_values_videos
            self.video_processor = VJEPA2VideoProcessor.from_pretrained(pretrained_model)
            self.processor_type = 'video'
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _to_absolute_path(self, path: str) -> str:
        """Convert relative path to absolute path using configured data directory."""
        if not Path(path).is_absolute():
            path_obj = Path(path)
            data_root = Path(self.cfg.path.data)
            if path_obj.parts[0] == "data":
                relative_path = Path(*path_obj.parts[1:])
                path = str(data_root / relative_path)
            else:
                path = str(data_root / path)
        return path
    
    def _load_embedding(self, csv_idx: int, agent_position: int) -> torch.Tensor:
        """Load pre-computed embedding from h5 file."""
        if self.embeddings_h5 is None:
            raise RuntimeError("Embeddings h5 file not loaded.")
        
        embedding = self.embeddings_h5[str(csv_idx)][str(agent_position)][:]
        return torch.from_numpy(embedding).float()
    
    def _load_video_clip(self, video_path: str, start_seconds: float, end_seconds: float) -> torch.Tensor:
        """Load video clip using decord."""
        expected_frames = int(self.cfg.data.fixed_duration_seconds * self.cfg.data.target_fps)
        video_full_path = self._to_absolute_path(video_path)
        
        try:
            decoder = VideoReader(video_full_path, ctx=cpu(0))
            video_fps = decoder.get_avg_fps()
            
            # Sample frames at target_fps
            timestamps = np.linspace(start_seconds, start_seconds + self.cfg.data.fixed_duration_seconds, 
                                     expected_frames, endpoint=False)
            
            # Apply time jitter if configured
            if self.cfg.data.time_jitter_max_seconds > 0:
                jitter = np.random.uniform(-self.cfg.data.time_jitter_max_seconds, 
                                           self.cfg.data.time_jitter_max_seconds, size=len(timestamps))
                timestamps = timestamps + jitter
                total_duration = len(decoder) / video_fps
                timestamps = np.clip(timestamps, 0, total_duration)
            
            frame_indices = (timestamps * video_fps).astype(int)
            max_frame_index = len(decoder) - 1
            frame_indices = np.clip(frame_indices, 0, max_frame_index)
            
            video_clip = decoder.get_batch(frame_indices.tolist())
            video_clip = torch.from_numpy(video_clip.asnumpy()).permute(0, 3, 1, 2).half()
            
            if self.cfg.data.mask_minimap:
                video_clip = apply_minimap_mask(video_clip)
            
            return video_clip
        except Exception as e:
            logger.warning(f"Failed to load video {video_path}: {e}, using placeholder")
            return torch.zeros(expected_frames, 3, 306, 544, dtype=torch.float16)
    
    def _transform_video(self, video_clip: torch.Tensor) -> torch.Tensor:
        """Transform video clip using the video processor.
        
        Args:
            video_clip: Video tensor of shape [T, C, H, W] in range [0, 255]
            
        Returns:
            Processed video tensor of shape [T, C, H, W] normalized for the model
        """
        # Convert from [T, C, H, W] to list of numpy arrays [H, W, C] in uint8
        num_frames = video_clip.shape[0]
        frames_list = []
        for i in range(num_frames):
            frame = video_clip[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            if frame.max() <= 1.0:
                frame = (frame * 255).astype('uint8')
            else:
                frame = frame.astype('uint8')
            frames_list.append(frame)
        
        if self.processor_type == 'video':
            # VJEPA2: uses videos= parameter and returns pixel_values_videos [1, T, C, H, W]
            video_processed = self.video_processor(videos=frames_list, return_tensors="pt")
            video_features = video_processed.pixel_values_videos.squeeze(0)  # [T, C, H, W]
        else:
            # Image-based processors: use images= parameter and return pixel_values
            processed = self.video_processor(images=frames_list, return_tensors="pt")
            video_features = processed.pixel_values
            # Video processors (vivit, videomae) return [1, T, C, H, W], need to squeeze
            # Image processors (siglip, clip, dinov2) return [T, C, H, W]
            if video_features.dim() == 5:
                video_features = video_features.squeeze(0)  # [T, C, H, W]
        
        return video_features
    
    def _construct_video_path(self, match_id: str, player_id: str, round_num: int) -> str:
        """Construct video path for a player's round."""
        video_folder = self.cfg.data.video_folder
        video_path = Path('data') / video_folder / str(match_id) / str(player_id) / f"round_{round_num}.mp4"
        return str(video_path)
    
    def _get_alive_agents(self, row: pd.Series) -> List[Dict]:
        """Get list of alive agents from row."""
        agents = []
        
        for i in range(5):
            agent_id = row.get(f'teammate_{i}_id', '')
            
            # Check if agent is valid (not empty)
            if not agent_id or str(agent_id).strip() == '' or str(agent_id) == 'nan':
                continue
            
            # Normalize agent_id to string for consistency (pandas may read as np.int64 or str)
            agent_id = str(agent_id)
            
            # Check health if available
            health = row.get(f'teammate_{i}_health', 100)
            if health == '' or (isinstance(health, (int, float)) and health <= 0):
                continue
            
            agents.append({
                'id': agent_id,
                'name': row.get(f'teammate_{i}_name', ''),
                'side': row.get(f'teammate_{i}_side', ''),
                'position': i,  # Store original position for embedding lookup
            })
        
        return agents
    
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
        row = self.df.iloc[idx]
        
        # Extract metadata
        start_seconds = row['start_seconds']
        end_seconds = row['end_seconds']
        match_id = row['match_id']
        round_num = row['round_num']
        pov_team_side = str(row['pov_team_side']).upper()
        original_csv_idx = row['idx']
        
        # Get alive agents
        alive_agents = self._get_alive_agents(row)
        
        if len(alive_agents) < self.cfg.data.min_agents:
            # Not enough agents - return minimal sample that will be masked
            return self._create_minimal_sample(pov_team_side, original_csv_idx)
        
        # Load videos
        agent_videos = []
        agent_ids = []
        
        for agent in alive_agents:
            video_path = self._construct_video_path(match_id, agent['id'], round_num)
            video_clip = self._load_video_clip(video_path, start_seconds, end_seconds)
            video_features = self._transform_video(video_clip)
            agent_videos.append(video_features)
            agent_ids.append(agent['id'])
        
        if len(agent_videos) < self.cfg.data.min_agents:
            return self._create_minimal_sample(pov_team_side, original_csv_idx)
        
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
    
    def _create_minimal_sample(self, pov_team_side: str, original_csv_idx: int) -> Dict:
        """Create a minimal sample for edge cases (not enough agents)."""
        pov_team_side_encoded = 1 if pov_team_side == 'CT' else 0
        
        num_frames = int(self.cfg.data.fixed_duration_seconds * self.cfg.data.target_fps)
        video = torch.zeros(1, num_frames, 3, 224, 224)
        
        return {
            'video': video,
            'num_agents': 0,  # Will be masked
            'pov_team_side': pov_team_side,
            'pov_team_side_encoded': torch.tensor(pov_team_side_encoded, dtype=torch.long),
            'agent_ids': [],
            'original_csv_idx': original_csv_idx,
        }


if __name__ == "__main__":
    print("ContrastiveDataset test placeholder")
