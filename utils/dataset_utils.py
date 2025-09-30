from pathlib import Path
import random
import json
from typing import List
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
    """
    video path example: /scratch/username/projects/CTFM/data/sample/recording/video/1-82e79d39-b8f2-482c-ab90-4941268a167b-1-1/76561198028656944/round_1.mp4
    metadata path example: /scratch/username/projects/CTFM/data/sample/recording/metadata/1-82e79d39-b8f2-482c-ab90-4941268a167b-1-1.json
    """
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


def get_team_voice_audio_clip_from_video_path(video_path: str) -> str:
    """
    team voice path example: /scratch/username/projects/CTFM/data/sample/recording/voice/1-82e79d39-b8f2-482c-ab90-4941268a167b-1-1/team_0/round_1.flac
    """
    team_number = get_player_team_number(video_path)
    
    video_path_obj = Path(video_path)
    
    match_id = video_path_obj.parent.parent.name
    round_filename = video_path_obj.stem  # filename without extension
    
    # Go three levels up: round file -> player -> match -> video directory
    voice_path = (
        video_path_obj.parent.parent.parent  # lands on "video_544x306"
        .parent  # lands on "recording"
        / "voice"
        / match_id
        / f"team_{team_number}"
        / f"{round_filename}.flac"
    )
    
    return str(voice_path)

def get_team_transcription_path_from_video_path(video_path: str, transcription_folder: str) -> str:
    """
    Get team transcription path from video path.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to team transcription file
        
    Example:
        Input: /scratch/username/projects/CTFM/data/sample/recording/video/1-82e79d39-b8f2-482c-ab90-4941268a167b-1-1/76561198028656944/round_1.mp4
        Output: /scratch/username/projects/CTFM/data/sample/recording/transcription/1-82e79d39-b8f2-482c-ab90-4941268a167b-1-1/team_0/round_1.json
    """
    team_number = get_player_team_number(video_path)
    
    video_path_obj = Path(video_path)
    player_id = video_path_obj.parent.name
    match_id = video_path_obj.parent.parent.name
    round_filename = video_path_obj.stem  # filename without extension
    
    transcription_path = video_path_obj.parent.parent.parent.parent / transcription_folder / match_id / f"team_{team_number}" / f"{round_filename}.json"
    return str(transcription_path)


def extract_text_segments_from_transcription(transcription_path: str, start_seconds: float, end_seconds: float, exclude_labels: List[str] = [], allow_overflow: bool = True) -> str:
    """
    Extract text segments from transcription file based on time range.
    
    Args:
        transcription_path: Path to the transcription JSON file
        start_seconds: Start time for the text clip
        end_seconds: End time for the text clip
        allow_overflow: If True, allow up to 1 segment overflow on each side
        
    Returns:
        Combined text from transcription segments that overlap with the specified time range
    """
    with open(transcription_path, "r") as f:
        transcription = json.load(f)
    
    chunks = transcription["chunks"]
    selected_segments = []
    
    for i, chunk in enumerate(chunks):
        timestamp = chunk["timestamp"]
        if len(timestamp) != 2:
            continue
            
        chunk_start, chunk_end = timestamp
        if chunk_start is None or chunk_end is None:
            continue
        
        if exclude_labels and "comm_type" in chunk and \
            any(comm_type in chunk["comm_type"] for comm_type in exclude_labels):
            continue
        
        # Check keys in priority order: enhanced_text, text_anonymized, text_de, text
        if "enhanced_text" in chunk:
            chunk_text = chunk["enhanced_text"]
        elif "text_anonymized" in chunk:
            chunk_text = chunk["text_anonymized"]
        elif "text_de" in chunk:
            chunk_text = chunk["text_de"]
        else:
            chunk_text = chunk["text"]
            
        if allow_overflow:
            # Include segments that overlap with the time range
            if chunk_start <= start_seconds <= chunk_end:
                selected_segments.append(chunk_text)
            elif chunk_start <= end_seconds <= chunk_end:
                selected_segments.append(chunk_text)
                
        # Include segments that are fully within the time range
        if start_seconds <= chunk_start and chunk_end <= end_seconds:
            selected_segments.append(chunk_text)
                
        # Early exit if we've passed the end time
        if end_seconds < chunk_start:
            break
    
    # Clean and join the text segments
    joined_text = " ".join(selected_segments).strip().replace("  ", " ").replace("\n", "").replace("\r", "")
    return joined_text if joined_text else ""


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