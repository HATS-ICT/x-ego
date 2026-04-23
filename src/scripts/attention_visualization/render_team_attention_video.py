"""
Render team-synchronized attention visualization video.

This script creates a grid visualization of all 5 agents from the same team:
- 3 rows: Original, Model + CECL, Model (off-the-shelf)
- 5 columns: One for each team member

Videos are synchronized and padded with black frames if a player dies early.

Usage:
    python -m src.scripts.attention_visualization.render_team_attention_video \
        --match_id "1-2b61eddf-3ab3-47ff-ac3e-3d730458667b-1-1" \
        --round_num 5 \
        --team_number 0 \
        --model_type dinov2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import platform
if platform.system() == 'Windows':
    import pathlib._local
    pathlib._local.PosixPath = pathlib._local.WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath

import argparse
import json
import torch
import numpy as np
import cv2
from decord import VideoReader, cpu
from tqdm import tqdm
from typing import List, Dict, Optional

from src.scripts.attention_visualization.render_attention_video import (
    load_video_frames,
    preprocess_frames_for_model,
    load_finetuned_vision_model,
    create_attention_overlay,
    add_title_bar_below,
    get_patch_dimensions,
    DEFAULT_EXPERIMENTS,
)
from src.scripts.attention_visualization.visualize_attention import (
    load_vision_model_eager,
    get_attention_weights,
)


# Model display names
MODEL_DISPLAY_NAMES = {
    'siglip2': 'SigLIP',
    'dinov2': 'DINOv2',
    'clip': 'CLIP',
    'vjepa2': 'V-JEPA2',
}


def get_team_players(metadata_path: Path, team_number: int) -> List[Dict]:
    """Get list of players for a specific team from metadata."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    team_players = [
        player for player in metadata['players'] 
        if player['team_number'] == team_number
    ]
    return team_players


def load_team_videos(
    video_base_path: Path,
    match_id: str,
    round_num: int,
    player_ids: List[str],
    apply_ui_mask: bool = True,
) -> tuple[List[Optional[np.ndarray]], float, int]:
    """
    Load videos for all team members.
    
    Returns:
        Tuple of (list of frame arrays, fps, max_frames)
        Frame arrays are None if video doesn't exist, otherwise [T, H, W, C]
    """
    videos = []
    fps = 4.0
    max_frames = 0
    
    for player_id in player_ids:
        video_path = video_base_path / match_id / player_id / f"round_{round_num}.mp4"
        
        if video_path.exists():
            frames, video_fps = load_video_frames(str(video_path), apply_ui_mask=apply_ui_mask)
            videos.append(frames)
            fps = video_fps
            max_frames = max(max_frames, len(frames))
        else:
            videos.append(None)
            print(f"  Warning: Video not found for player {player_id}")
    
    return videos, fps, max_frames


def pad_video_to_length(frames: Optional[np.ndarray], target_length: int, frame_shape: tuple) -> np.ndarray:
    """Pad video with black frames to reach target length."""
    if frames is None:
        # Return all black frames
        return np.zeros((target_length, *frame_shape), dtype=np.uint8)
    
    current_length = len(frames)
    if current_length >= target_length:
        return frames[:target_length]
    
    # Pad with black frames
    padding = np.zeros((target_length - current_length, *frame_shape), dtype=np.uint8)
    return np.vstack([frames, padding])


def render_team_attention_video(
    match_id: str,
    round_num: int,
    team_number: int,
    model_type: str,
    experiment_name: str,
    epoch: int,
    output_path: str,
    video_base_path: str = "data/video_306x306_4fps",
    metadata_base_path: str = "data/metadata",
    batch_size: int = 8,
    show_titles: bool = True,
) -> None:
    """
    Render team-synchronized attention visualization video.
    
    Grid layout:
    - 3 rows: Original, Model + CECL, Model
    - 5 columns: One per team member
    """
    video_base = Path(video_base_path)
    metadata_base = Path(metadata_base_path)
    
    # Get team players from metadata
    metadata_path = metadata_base / f"{match_id}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    team_players = get_team_players(metadata_path, team_number)
    player_ids = [p['steamid'] for p in team_players]
    player_names = [p['name'] for p in team_players]
    
    print(f"Team {team_number} players: {player_names}")
    
    if len(player_ids) != 5:
        print(f"Warning: Expected 5 players, found {len(player_ids)}")
    
    # Pad to 5 players if needed
    while len(player_ids) < 5:
        player_ids.append(None)
        player_names.append("N/A")
    
    # Load all team videos
    print("Loading team videos...")
    videos, fps, max_frames = load_team_videos(video_base, match_id, round_num, player_ids[:5])
    
    if max_frames == 0:
        raise ValueError("No valid videos found for this team/round")
    
    print(f"Max frames: {max_frames}, FPS: {fps}")
    
    # Get frame shape from first valid video
    frame_shape = None
    for v in videos:
        if v is not None:
            frame_shape = v.shape[1:]  # (H, W, C)
            break
    
    if frame_shape is None:
        raise ValueError("Could not determine frame shape")
    
    # Pad all videos to same length
    videos = [pad_video_to_length(v, max_frames, frame_shape) for v in videos]
    
    num_patches_h, num_patches_w = get_patch_dimensions(model_type)
    display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type.upper())
    
    # Load models
    print(f"Loading pretrained {model_type} model...")
    pretrained_model = load_vision_model_eager(model_type)
    pretrained_model.eval()
    pretrained_model.cuda()
    
    print(f"Loading finetuned {model_type} model from {experiment_name}...")
    finetuned_model = load_finetuned_vision_model(experiment_name, epoch, model_type)
    
    # Prepare output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    h, w, c = frame_shape
    title_bar_height = 30 if show_titles else 0
    
    # Grid: 5 columns x 3 rows
    out_width = w * 5
    out_height = (h + title_bar_height) * 3
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
    
    # Row labels
    row_labels = ["Original", f"{display_name} + CECL", display_name]
    
    print("Processing frames...")
    
    # Process in batches
    for batch_start in tqdm(range(0, max_frames, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, max_frames)
        batch_size_actual = batch_end - batch_start
        
        # Compute attention for each player's batch
        player_finetuned_attentions = []
        player_pretrained_attentions = []
        
        for player_idx in range(5):
            batch_frames = videos[player_idx][batch_start:batch_end]
            
            # Check if all frames are black (player dead/no video)
            if np.max(batch_frames) == 0:
                player_finetuned_attentions.append(None)
                player_pretrained_attentions.append(None)
                continue
            
            video_tensor = preprocess_frames_for_model(batch_frames, model_type)
            video_tensor = video_tensor.unsqueeze(0).cuda()
            
            with torch.no_grad():
                finetuned_attn = get_attention_weights(finetuned_model, video_tensor, model_type)
                pretrained_attn = get_attention_weights(pretrained_model, video_tensor, model_type)
            
            player_finetuned_attentions.append(finetuned_attn)
            player_pretrained_attentions.append(pretrained_attn)
        
        # Render each frame in batch
        for i in range(batch_size_actual):
            frame_idx = batch_start + i
            
            # Build the 3x5 grid
            grid_rows = []
            
            for row_idx in range(3):  # Original, CECL, Pretrained
                row_frames = []
                
                for player_idx in range(5):
                    frame = videos[player_idx][frame_idx]
                    
                    if row_idx == 0:
                        # Original
                        display_frame = frame
                    elif row_idx == 1:
                        # CECL (finetuned)
                        if player_finetuned_attentions[player_idx] is not None:
                            attention = player_finetuned_attentions[player_idx][i]
                            display_frame = create_attention_overlay(
                                frame, attention, model_type,
                                num_patches_h, num_patches_w,
                            )
                        else:
                            display_frame = frame  # Black frame stays black
                    else:
                        # Pretrained
                        if player_pretrained_attentions[player_idx] is not None:
                            attention = player_pretrained_attentions[player_idx][i]
                            display_frame = create_attention_overlay(
                                frame, attention, model_type,
                                num_patches_h, num_patches_w,
                            )
                        else:
                            display_frame = frame
                    
                    # Add title for first frame row only (player names)
                    if show_titles:
                        if row_idx == 0:
                            # Show player name on original row
                            title = player_names[player_idx][:12]  # Truncate long names
                        else:
                            title = row_labels[row_idx]
                        display_frame = add_title_bar_below(display_frame, title, title_bar_height)
                    
                    row_frames.append(display_frame)
                
                grid_rows.append(np.hstack(row_frames))
            
            combined = np.vstack(grid_rows)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            writer.write(combined_bgr)
    
    writer.release()
    print(f"Saved team attention video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render team-synchronized attention visualization video"
    )
    parser.add_argument(
        '--match_id',
        type=str,
        required=True,
        help='Match ID (folder name in video directory)'
    )
    parser.add_argument(
        '--round_num',
        type=int,
        required=True,
        help='Round number to visualize'
    )
    parser.add_argument(
        '--team_number',
        type=int,
        default=0,
        choices=[0, 1],
        help='Team number (0 or 1)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='dinov2',
        choices=['dinov2', 'siglip2', 'clip', 'vjepa2'],
        help='Type of vision model to use'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name for finetuned checkpoint'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=39,
        help='Epoch of checkpoint to load'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to output video'
    )
    parser.add_argument(
        '--video_base_path',
        type=str,
        default="data/video_306x306_4fps",
        help='Base path for videos'
    )
    parser.add_argument(
        '--metadata_base_path',
        type=str,
        default="data/metadata",
        help='Base path for metadata'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for frame processing'
    )
    parser.add_argument(
        '--no_titles',
        action='store_true',
        help='Disable title text on frames'
    )
    
    args = parser.parse_args()
    
    # Use default experiment if not specified
    if args.experiment_name is None:
        args.experiment_name = DEFAULT_EXPERIMENTS[args.model_type]
        print(f"Using default experiment: {args.experiment_name}")
    
    # Generate default output path if not specified
    if args.output_path is None:
        args.output_path = f"artifacts/attention_videos_team/{args.model_type}_{args.match_id}_round{args.round_num}_team{args.team_number}.mp4"
    
    render_team_attention_video(
        match_id=args.match_id,
        round_num=args.round_num,
        team_number=args.team_number,
        model_type=args.model_type,
        experiment_name=args.experiment_name,
        epoch=args.epoch,
        output_path=args.output_path,
        video_base_path=args.video_base_path,
        metadata_base_path=args.metadata_base_path,
        batch_size=args.batch_size,
        show_titles=not args.no_titles,
    )


if __name__ == "__main__":
    main()
