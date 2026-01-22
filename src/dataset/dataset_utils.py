from pathlib import Path
import random
import json
import torch
import numpy as np
from typing import Tuple, Any, Dict
from decord import VideoReader, cpu
import logging

logger = logging.getLogger(__name__)


def to_absolute_path(path: str) -> str:
    if not Path(path).is_absolute():
        path = str(Path(__file__).parent.parent.parent.parent / path)
    return path

def get_random_segment(full_duration, fixed_segment_duration):
    """
    Get a random segment with fixed duration.
    
    Args:
        full_duration: Total duration of the video
        fixed_segment_duration: Fixed duration for the segment
        
    Returns:
        tuple: (start_seconds, end_seconds) where end_seconds - start_seconds == fixed_segment_duration
               If video is shorter than fixed duration, returns (0, full_duration)
    """
    if full_duration <= fixed_segment_duration:
        # Video is shorter than or equal to fixed duration, use entire video
        return 0.0, full_duration
    else:
        # Video is longer, sample a random segment of fixed duration
        max_start = full_duration - fixed_segment_duration
        start = random.uniform(0, max_start)
        end = start + fixed_segment_duration
        return start, end


def get_player_team_number(video_path: str) -> int:
    video_path_obj = Path(video_path)
    player_id = video_path_obj.parent.name
    match_id = video_path_obj.parent.parent.name
    metadata_path = video_path_obj.parent.parent.parent.parent / "metadata" / f"{match_id}.json"
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    # TODO: could be slightly optimized by using a dictionary
    for player in metadata["players"]:
        if player["steamid"] == player_id:
            return player["team_number"]
    return None


def apply_minimap_mask(video_clip: torch.Tensor) -> torch.Tensor:
    num_frames, channels, height, width = video_clip.shape
    
    mask_width = width // 5
    mask_height = height * 3 // 10
    
    video_clip[:, :, :mask_height, :mask_width] = 0
    
    return video_clip


def init_video_processor(cfg: Dict) -> Tuple[Any, str]:
    """
    Initialize video processor based on model type from config.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (video_processor, processor_type)
        processor_type: 'image' uses images= param and returns pixel_values
                       'video' uses videos= param and returns pixel_values_videos
    """
    from transformers import AutoImageProcessor, VivitImageProcessor, VideoMAEImageProcessor, VJEPA2VideoProcessor
    from ..models.modules.video_encoder import MODEL_TYPE_TO_PRETRAINED
    
    model_type = cfg.model.encoder.video.model_type
    pretrained_model = MODEL_TYPE_TO_PRETRAINED[model_type]
    
    # Different models need different processors and have different output formats
    if model_type in ['siglip', 'siglip2', 'clip', 'dinov2']:
        # Image-based models: AutoImageProcessor, returns pixel_values
        video_processor = AutoImageProcessor.from_pretrained(pretrained_model)
        processor_type = 'image'
    elif model_type == 'vivit':
        # ViViT: specific image processor, returns pixel_values
        video_processor = VivitImageProcessor.from_pretrained(pretrained_model)
        processor_type = 'image'
    elif model_type == 'videomae':
        # VideoMAE: specific image processor, returns pixel_values
        video_processor = VideoMAEImageProcessor.from_pretrained(pretrained_model)
        processor_type = 'image'
    elif model_type == 'vjepa2':
        # VJEPA2: video processor, returns pixel_values_videos
        video_processor = VJEPA2VideoProcessor.from_pretrained(pretrained_model)
        processor_type = 'video'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return video_processor, processor_type


def construct_video_path(cfg: Dict, match_id: str, player_id: str, round_num: int) -> str:
    """Construct video path for a player's round."""
    video_folder = cfg.data.video_folder
    data_root = Path(cfg.path.data)
    video_path = data_root / video_folder / str(match_id) / str(player_id) / f"round_{round_num}.mp4"
    return str(video_path)


def load_video_clip(cfg: Dict, video_full_path: str, start_seconds: float, end_seconds: float) -> torch.Tensor:
    """
    Load video clip using decord.
    
    Args:
        cfg: Configuration dictionary
        video_path: Path to the video file
        start_seconds: Start time of the clip
        end_seconds: End time of the clip (not used, kept for compatibility)
        
    Returns:
        Video tensor of shape (num_frames, channels, height, width)
    """
    expected_frames = int(cfg.data.fixed_duration_seconds * cfg.data.target_fps)
    
    try:
        decoder = VideoReader(video_full_path, ctx=cpu(0))
        video_fps = decoder.get_avg_fps()
        
        # Sample frames at target_fps
        timestamps = np.linspace(start_seconds, start_seconds + cfg.data.fixed_duration_seconds, 
                                 expected_frames, endpoint=False)
        
        # Apply time jitter if configured
        if cfg.data.time_jitter_max_seconds > 0:
            jitter = np.random.uniform(-cfg.data.time_jitter_max_seconds, 
                                       cfg.data.time_jitter_max_seconds, size=len(timestamps))
            timestamps = timestamps + jitter
            total_duration = len(decoder) / video_fps
            timestamps = np.clip(timestamps, 0, total_duration)
        
        frame_indices = (timestamps * video_fps).astype(int)
        max_frame_index = len(decoder) - 1
        frame_indices = np.clip(frame_indices, 0, max_frame_index)
        
        video_clip = decoder.get_batch(frame_indices.tolist())
        video_clip = torch.from_numpy(video_clip.asnumpy()).permute(0, 3, 1, 2).half()
        
        # Apply UI masking based on configuration
        ui_mask = getattr(cfg.data, 'ui_mask', 'none')
        
        if ui_mask == 'minimap_only':
            video_clip = apply_minimap_mask(video_clip)
        elif ui_mask == 'all':
            # TODO: Implement full UI masking
            raise NotImplementedError("ui_mask='all' is not yet implemented")
        # elif ui_mask == 'none': no masking applied
        
        return video_clip
    except Exception as e:
        logger.warning(f"Failed to load video {video_full_path}: {e}, using placeholder")
        return torch.zeros(expected_frames, 3, 306, 544, dtype=torch.float16)


def transform_video(video_processor: Any, processor_type: str, video_clip: torch.Tensor) -> torch.Tensor:
    """
    Transform video clip using the video processor.
    
    Args:
        video_processor: The video processor instance
        processor_type: 'image' or 'video'
        video_clip: Video tensor of shape [T, C, H, W] in range [0, 255]
        
    Returns:
        Processed video tensor of shape [T, C, H, W] normalized for the model
    """
    if processor_type == 'video':
        # VJEPA2: uses videos= parameter and returns pixel_values_videos [1, T, C, H, W]
        video_processed = video_processor(videos=video_clip, return_tensors="pt")
        video_features = video_processed.pixel_values_videos.squeeze(0)  # [T, C, H, W]
    else:
        # Image-based processors: use images= parameter and return pixel_values
        processed = video_processor(images=video_clip, return_tensors="pt")
        video_features = processed.pixel_values
        # Video processors (vivit, videomae) return [1, T, C, H, W], need to squeeze
        # Image processors (siglip, clip, dinov2) return [T, C, H, W]
        if video_features.dim() == 5:
            video_features = video_features.squeeze(0)  # [T, C, H, W]
    
    return video_features