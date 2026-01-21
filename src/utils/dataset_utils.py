from pathlib import Path
import random
import json
import torch


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
    """
    Apply a black mask to the top-left corner of video frames to hide the minimap.
    
    Args:
        video_clip: Video tensor of shape (num_frames, channels, height, width)
        
    Returns:
        Video tensor with minimap masked (same shape as input)
    """
    # Get video dimensions
    num_frames, channels, height, width = video_clip.shape
    
    # Calculate mask dimensions (1/4 of width and height)
    mask_width = width // 5
    mask_height = height * 3 // 10
    
    # Apply black mask to top-left corner for all frames and channels
    video_clip[:, :, :mask_height, :mask_width] = 0
    
    return video_clip