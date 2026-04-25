from pathlib import Path
import random
import json
from functools import lru_cache
import torch
import numpy as np
from typing import Tuple, Any, Dict
from decord import VideoReader, cpu

UI_MASK_PATH = Path(__file__).resolve().parent.parent / "assets" / "ui_mask.json"
DEFAULT_UI_MASK_RESOLUTION = "306x306"


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


@lru_cache(maxsize=1)
def load_ui_mask_boxes() -> dict:
    with open(UI_MASK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["ui_mask"]


def get_ui_mask_boxes_for_resolution(width: int, height: int) -> list[dict]:
    masks_by_resolution = load_ui_mask_boxes()
    resolution_key = f"{width}x{height}"
    if resolution_key in masks_by_resolution:
        return masks_by_resolution[resolution_key]

    source_key = DEFAULT_UI_MASK_RESOLUTION
    if source_key not in masks_by_resolution:
        raise KeyError(f"{UI_MASK_PATH} does not contain a {source_key} UI mask")

    source_width, source_height = (int(value) for value in source_key.split("x"))
    scale_x = width / source_width
    scale_y = height / source_height
    return [
        {
            "x": box["x"] * scale_x,
            "y": box["y"] * scale_y,
            "w": box["w"] * scale_x,
            "h": box["h"] * scale_y,
        }
        for box in masks_by_resolution[source_key]
    ]


def apply_all_ui_mask(video_clip: torch.Tensor) -> torch.Tensor:
    _, _, height, width = video_clip.shape

    for box in get_ui_mask_boxes_for_resolution(width, height):
        x1 = max(0, min(width, int(round(box["x"]))))
        y1 = max(0, min(height, int(round(box["y"]))))
        x2 = max(0, min(width, int(round(box["x"] + box["w"]))))
        y2 = max(0, min(height, int(round(box["y"] + box["h"]))))
        if x2 > x1 and y2 > y1:
            video_clip[:, :, y1:y2, x1:x2] = 0
    
    return video_clip


def apply_random_tube_mask(
    video_clip: torch.Tensor,
    num_tubes: int,
    min_size_ratio: float,
    max_size_ratio: float,
) -> torch.Tensor:
    """
    Apply random tube masks to the video clip.
    
    A tube mask is a rectangular region that is masked consistently across all frames,
    creating a "tube" through the temporal dimension. This encourages the model to
    learn from context rather than relying on specific spatial regions.
    
    Args:
        video_clip: Video tensor of shape [T, C, H, W]
        num_tubes: Number of random tube masks to apply
        min_size_ratio: Minimum size of each tube as a ratio of frame dimensions
        max_size_ratio: Maximum size of each tube as a ratio of frame dimensions
        
    Returns:
        Masked video tensor of shape [T, C, H, W]
    """
    num_frames, channels, height, width = video_clip.shape
    
    for _ in range(num_tubes):
        # Random tube dimensions
        tube_width = random.randint(int(width * min_size_ratio), int(width * max_size_ratio))
        tube_height = random.randint(int(height * min_size_ratio), int(height * max_size_ratio))
        
        # Random tube position (ensure it fits within frame)
        tube_x = random.randint(0, width - tube_width)
        tube_y = random.randint(0, height - tube_height)
        
        # Apply mask across all frames (tube through time)
        video_clip[:, :, tube_y:tube_y + tube_height, tube_x:tube_x + tube_width] = 0
    
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
                       'torchvision_image' applies tensor transforms locally
    """
    from transformers import AutoImageProcessor, VivitImageProcessor, VideoMAEImageProcessor, VJEPA2VideoProcessor
    from ..models.modules.video_encoder import MODEL_TYPE_TO_PRETRAINED, normalize_model_type
    
    model_type = normalize_model_type(cfg.model.encoder.model_type)
    cfg.model.encoder.model_type = model_type
    pretrained_model = MODEL_TYPE_TO_PRETRAINED[model_type]
    
    # Different models need different processors and have different output formats
    if model_type in ['siglip2', 'clip', 'dinov2', 'dinov3']:
        # Image-based models: AutoImageProcessor, returns pixel_values
        video_processor = AutoImageProcessor.from_pretrained(pretrained_model, use_fast=True)
        processor_type = 'image'
    elif model_type == 'vivit':
        # ViViT: expects list of HWC numpy arrays, returns pixel_values
        video_processor = VivitImageProcessor.from_pretrained(pretrained_model)
        processor_type = 'video_frames'
    elif model_type == 'videomae':
        # VideoMAE: expects list of HWC numpy arrays, returns pixel_values
        video_processor = VideoMAEImageProcessor.from_pretrained(pretrained_model)
        processor_type = 'video_frames'
    elif model_type == 'vjepa2':
        # VJEPA2: video processor, returns pixel_values_videos
        video_processor = VJEPA2VideoProcessor.from_pretrained(pretrained_model)
        processor_type = 'video'
    elif model_type == 'resnet50':
        # Torchvision ResNet-50 uses standard ImageNet tensor preprocessing.
        video_processor = {
            "size": 224,
            "mean": torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            "std": torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        }
        processor_type = 'torchvision_image'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return video_processor, processor_type


def construct_video_path(cfg: Dict, match_id: str, player_id: str, round_num: int) -> str:
    """Construct video path for a player's round."""
    video_folder = cfg.data.video_folder
    data_root = Path(cfg.path.data)
    
    map_name = cfg.data.map
    video_path = data_root / map_name / video_folder / str(match_id) / str(player_id) / f"round_{round_num}.mp4"
    
    return str(video_path)


def load_video_clip(cfg: Dict, video_full_path: str, start_seconds: float, end_seconds: float) -> Dict[str, torch.Tensor]:
    """
    Load video clip using decord.
    
    Args:
        cfg: Configuration dictionary
        video_path: Path to the video file
        start_seconds: Start time of the clip
        end_seconds: End time of the clip (not used, kept for compatibility)
        
    Returns:
        Dictionary containing:
            - 'video': Video tensor of shape (num_frames, channels, height, width)
    """
    expected_frames = int(cfg.data.fixed_duration_seconds * cfg.data.target_fps)

    if not Path(video_full_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_full_path}")

    decoder = VideoReader(video_full_path, ctx=cpu(0))
    video_fps = decoder.get_avg_fps()

    # Sample frames at target_fps.
    timestamps = np.linspace(
        start_seconds,
        start_seconds + cfg.data.fixed_duration_seconds,
        expected_frames,
        endpoint=False,
    )

    if cfg.data.time_jitter_max_seconds > 0:
        jitter = np.random.uniform(
            -cfg.data.time_jitter_max_seconds,
            cfg.data.time_jitter_max_seconds,
            size=len(timestamps),
        )
        timestamps = timestamps + jitter
        total_duration = len(decoder) / video_fps
        timestamps = np.clip(timestamps, 0, total_duration)

    frame_indices = (timestamps * video_fps).astype(int)
    max_frame_index = len(decoder) - 1
    frame_indices = np.clip(frame_indices, 0, max_frame_index)

    video_clip = decoder.get_batch(frame_indices.tolist())
    video_clip = torch.from_numpy(video_clip.asnumpy()).permute(0, 3, 1, 2).half()

    ui_mask = cfg.data.ui_mask
    if ui_mask == "minimap_only":
        video_clip = apply_minimap_mask(video_clip)
    elif ui_mask == "all":
        video_clip = apply_all_ui_mask(video_clip)
    elif ui_mask != "none":
        raise ValueError(f"Unsupported data.ui_mask: {ui_mask}")

    result = {"video": video_clip}

    if cfg.data.random_mask.enable:
        result["video"] = apply_random_tube_mask(
            video_clip,
            num_tubes=cfg.data.random_mask.num_tubes,
            min_size_ratio=cfg.data.random_mask.min_size_ratio,
            max_size_ratio=cfg.data.random_mask.max_size_ratio,
        )

    return result


def transform_video(video_processor: Any, processor_type: str, video_clip: torch.Tensor) -> torch.Tensor:
    """
    Transform video clip using the video processor.
    
    Args:
        video_processor: The video processor instance
        processor_type: 'image', 'video', 'video_frames', or 'torchvision_image'
        video_clip: Video tensor of shape [T, C, H, W] in range [0, 255]
        
    Returns:
        Processed video tensor of shape [T, C, H, W] normalized for the model
    """
    if processor_type == 'video':
        # VJEPA2: uses videos= parameter and returns pixel_values_videos [1, T, C, H, W]
        video_processed = video_processor(videos=video_clip, return_tensors="pt")
        video_features = video_processed.pixel_values_videos.squeeze(0)  # [T, C, H, W]
    elif processor_type == 'video_frames':
        # VivitImageProcessor/VideoMAEImageProcessor: expects list of HWC numpy arrays
        # Convert from [T, C, H, W] tensor to list of [H, W, C] numpy arrays
        frames_list = [frame.permute(1, 2, 0).numpy().astype(np.uint8) for frame in video_clip]
        processed = video_processor(images=frames_list, return_tensors="pt")
        video_features = processed.pixel_values
        # Returns [1, T, C, H, W], need to squeeze
        if video_features.dim() == 5:
            video_features = video_features.squeeze(0)  # [T, C, H, W]
    elif processor_type == 'image':
        # Image-based processors (siglip, clip, dinov2): use images= parameter
        processed = video_processor(images=video_clip, return_tensors="pt")
        video_features = processed.pixel_values
        # Returns [T, C, H, W]
    elif processor_type == 'torchvision_image':
        video_features = video_clip.float() / 255.0
        video_features = torch.nn.functional.interpolate(
            video_features,
            size=(video_processor["size"], video_processor["size"]),
            mode="bilinear",
            align_corners=False,
        )
        mean = video_processor["mean"].to(device=video_features.device, dtype=video_features.dtype)
        std = video_processor["std"].to(device=video_features.device, dtype=video_features.dtype)
        video_features = (video_features - mean) / std
    else:
        raise ValueError(f"Unknown processor_type: {processor_type}")
    
    return video_features
