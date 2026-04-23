"""
Batch render team-synchronized attention videos for multiple models.

Randomly samples k match/round/team combinations and renders for each model type.

Usage:
    python -m src.scripts.attention_visualization.batch_render_team_videos
    python -m src.scripts.attention_visualization.batch_render_team_videos --num_samples 5 --models dinov2 siglip2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import random
import json
from typing import List, Tuple

from src.scripts.attention_visualization.render_team_attention_video import (
    render_team_attention_video,
    DEFAULT_EXPERIMENTS,
)


def get_valid_team_samples(
    video_base_path: Path,
    metadata_base_path: Path,
) -> List[Tuple[str, int, int]]:
    """
    Find all valid (match_id, round_num, team_number) combinations.
    
    Returns list of tuples: (match_id, round_num, team_number)
    """
    samples = []
    
    # Iterate through match folders
    for match_dir in video_base_path.iterdir():
        if not match_dir.is_dir():
            continue
        
        match_id = match_dir.name
        metadata_path = metadata_base_path / f"{match_id}.json"
        
        if not metadata_path.exists():
            continue
        
        # Load metadata to get team info
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get players by team
        team_0_players = [p['steamid'] for p in metadata['players'] if p['team_number'] == 0]
        team_1_players = [p['steamid'] for p in metadata['players'] if p['team_number'] == 1]
        
        # Find rounds that have videos for at least 3 players per team
        player_dirs = list(match_dir.iterdir())
        
        # Collect all available rounds
        all_rounds = set()
        for player_dir in player_dirs:
            if player_dir.is_dir():
                for video_file in player_dir.glob("round_*.mp4"):
                    round_num = int(video_file.stem.split('_')[1])
                    all_rounds.add(round_num)
        
        # Check each round for team coverage
        for round_num in all_rounds:
            # Check team 0
            team_0_videos = sum(
                1 for pid in team_0_players 
                if (match_dir / pid / f"round_{round_num}.mp4").exists()
            )
            if team_0_videos >= 3:
                samples.append((match_id, round_num, 0))
            
            # Check team 1
            team_1_videos = sum(
                1 for pid in team_1_players 
                if (match_dir / pid / f"round_{round_num}.mp4").exists()
            )
            if team_1_videos >= 3:
                samples.append((match_id, round_num, 1))
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Batch render team-synchronized attention videos"
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of random team samples to process per model'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['dinov2', 'siglip2', 'clip'],
        choices=['dinov2', 'siglip2', 'clip', 'vjepa2'],
        help='Model types to process'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=39,
        help='Epoch of checkpoint to load'
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
        '--output_dir',
        type=str,
        default="artifacts/attention_videos_team_batch",
        help='Output directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for frame processing'
    )
    
    args = parser.parse_args()
    
    video_base = Path(args.video_base_path)
    metadata_base = Path(args.metadata_base_path)
    output_dir = Path(args.output_dir)
    
    # Find valid samples
    print("Scanning for valid team samples...")
    all_samples = get_valid_team_samples(video_base, metadata_base)
    print(f"Found {len(all_samples)} valid team samples")
    
    if len(all_samples) == 0:
        print("No valid samples found!")
        return
    
    # Sample randomly (no seed for true randomness)
    sampled = random.sample(all_samples, min(args.num_samples, len(all_samples)))
    print(f"Selected {len(sampled)} samples:")
    for match_id, round_num, team_num in sampled:
        print(f"  - {match_id} round {round_num} team {team_num}")
    
    # Process each model
    for model_type in args.models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_type}")
        print(f"{'='*60}")
        
        experiment_name = DEFAULT_EXPERIMENTS[model_type]
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (match_id, round_num, team_num) in enumerate(sampled):
            print(f"\n[{i+1}/{len(sampled)}] {match_id} round {round_num} team {team_num}")
            
            output_filename = f"{match_id}_round{round_num}_team{team_num}.mp4"
            output_path = model_output_dir / output_filename
            
            if output_path.exists():
                print(f"  Skipping (already exists)")
                continue
            
            try:
                render_team_attention_video(
                    match_id=match_id,
                    round_num=round_num,
                    team_number=team_num,
                    model_type=model_type,
                    experiment_name=experiment_name,
                    epoch=args.epoch,
                    output_path=str(output_path),
                    video_base_path=args.video_base_path,
                    metadata_base_path=args.metadata_base_path,
                    batch_size=args.batch_size,
                    show_titles=True,
                )
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
