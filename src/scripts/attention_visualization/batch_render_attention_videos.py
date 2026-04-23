"""
Batch render attention visualization videos for all model types.

This script runs render_attention_video for all 4 model types (dinov2, siglip2, clip, vjepa2)
on 20 randomly sampled videos.

Usage:
    python -m src.scripts.attention_visualization.batch_render_attention_videos
    python -m src.scripts.attention_visualization.batch_render_attention_videos --num_videos 10
    python -m src.scripts.attention_visualization.batch_render_attention_videos --models dinov2 siglip2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import random
from typing import List

from src.scripts.attention_visualization.render_attention_video import (
    render_attention_video,
    DEFAULT_EXPERIMENTS,
)


def get_all_video_paths(video_base_path: Path) -> List[Path]:
    """Get all video paths from the video directory."""
    video_paths = list(video_base_path.glob("*/*/*.mp4"))
    return video_paths


def main():
    parser = argparse.ArgumentParser(
        description="Batch render attention videos for multiple models"
    )
    parser.add_argument(
        '--video_base_path',
        type=str,
        default="data/video_306x306_4fps",
        help='Base path containing video folders'
    )
    parser.add_argument(
        '--num_videos',
        type=int,
        default=20,
        help='Number of random videos to process per model'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['dinov2', 'siglip2', 'clip'],
        choices=['dinov2', 'siglip2', 'clip', 'vjepa2'],
        help='Model types to process (default: dinov2, siglip2, clip)'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=39,
        help='Epoch of checkpoint to load'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for video sampling'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="artifacts/attention_videos_batch",
        help='Output directory for rendered videos'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for frame processing'
    )
    
    args = parser.parse_args()
    
    # Get all video paths
    video_base_path = Path(args.video_base_path)
    all_videos = get_all_video_paths(video_base_path)
    print(f"Found {len(all_videos)} total videos in {video_base_path}")
    
    # Sample random videos
    random.seed(args.seed)
    sampled_videos = random.sample(all_videos, min(args.num_videos, len(all_videos)))
    print(f"Sampled {len(sampled_videos)} videos for processing")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each model type
    for model_type in args.models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_type}")
        print(f"{'='*60}")
        
        experiment_name = DEFAULT_EXPERIMENTS[model_type]
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, video_path in enumerate(sampled_videos):
            print(f"\n[{i+1}/{len(sampled_videos)}] Processing: {video_path.name}")
            
            # Create output filename from video path structure
            # e.g., match_id/player_id/round_X.mp4 -> match_id_player_id_round_X.mp4
            match_id = video_path.parent.parent.name
            player_id = video_path.parent.name
            video_name = video_path.stem
            output_filename = f"{match_id}_{player_id}_{video_name}.mp4"
            output_path = model_output_dir / output_filename
            
            # Skip if already exists
            if output_path.exists():
                print(f"  Skipping (already exists): {output_path}")
                continue
            
            try:
                render_attention_video(
                    video_path=str(video_path),
                    model_type=model_type,
                    experiment_name=experiment_name,
                    epoch=args.epoch,
                    output_path=str(output_path),
                    batch_size=args.batch_size,
                    show_titles=True,
                )
            except Exception as e:
                print(f"  Error processing {video_path}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
