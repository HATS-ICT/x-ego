import csv
from pathlib import Path
import glob
import os
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Split configuration
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Get paths from environment variables
DATA_BASE_PATH = Path(os.getenv('DATA_BASE_PATH', 'data'))
if not DATA_BASE_PATH.is_absolute():
    # If relative path, make it relative to script location
    DATA_BASE_PATH = Path(__file__).resolve().parent.parent / DATA_BASE_PATH

SRC_BASE_PATH = Path(os.getenv('SRC_BASE_PATH', str(Path(__file__).resolve().parent.parent)))
if not SRC_BASE_PATH.is_absolute():
    SRC_BASE_PATH = Path(__file__).resolve().parent.parent / SRC_BASE_PATH

# Directory configuration
VIDEO_DIR = "video_544x306_30fps"
OUTPUT_FILE = "match_round_partitioned.csv"


def get_video_list_from_dir(dir_path):
    """Get all video paths from a directory."""
    # Use pathlib for cleaner path handling
    dir_path = Path(dir_path)
    video_paths = list(dir_path.rglob("*.mp4"))  # Recursive glob for .mp4 files
    return [str(path) for path in video_paths]  # Convert back to strings for compatibility


def extract_path_components(video_path):
    """Extract match, player, and round information from video path."""
    try:
        # Convert to Path object for easier manipulation
        path = Path(video_path)
        
        # Get relative path from DATA_BASE_PATH
        rel_path = path.relative_to(DATA_BASE_PATH)
        
        # Remove the video directory prefix using pathlib
        if rel_path.parts[0] == VIDEO_DIR:
            # Skip the first part (video_544x306_30fps)
            remaining_parts = rel_path.parts[1:]
        else:
            remaining_parts = rel_path.parts
        
        # Extract components: match_id/player_id/round_x.mp4
        if len(remaining_parts) >= 1:
            match_id = remaining_parts[0]
        else:
            return None, None, None
            
        player_id = remaining_parts[1] if len(remaining_parts) >= 2 else None
        
        # Extract round info from filename (e.g., "round_18.mp4" -> "round_18")
        if len(remaining_parts) >= 3:
            filename = remaining_parts[2]
            round_info = Path(filename).stem  # Automatically removes extension
        else:
            round_info = None
        
        return match_id, player_id, round_info
    except Exception as e:
        print(f"Error processing path {video_path}: {e}")
        return None, None, None


# Removed exclusion and required split functions - using simple random partitioning


def assign_random_split():
    """Randomly assign a split based on the configured ratios."""
    rand_val = random.random()
    if rand_val < TRAIN_RATIO:
        return "train"
    elif rand_val < TRAIN_RATIO + VAL_RATIO:
        return "val"
    else:
        return "test"


def get_all_match_round_combinations():
    """Get all unique match_id and round_number combinations from the video directory."""
    match_round_combinations = set()
    
    # Look directly in the VIDEO_DIR
    video_dir_path = DATA_BASE_PATH / VIDEO_DIR
    
    if video_dir_path.exists():
        paths = get_video_list_from_dir(video_dir_path)
        
        for path in paths:
            match_id, player_id, round_info = extract_path_components(path)
            
            if match_id and round_info:
                # Extract round number from round_info (e.g., "round_1" -> "1")
                try:
                    round_number = int(round_info.split('_')[-1]) if 'round_' in round_info else int(round_info)
                    match_round_combinations.add((match_id, round_number))
                except ValueError:
                    print(f"Warning: Could not extract round number from {round_info}")
    else:
        print(f"Warning: Video directory {video_dir_path} does not exist")
    
    return list(match_round_combinations)


def build_data_partition_csv():
    """Create partitioned CSV file with train/val/test splits by match and round."""
    output_path = DATA_BASE_PATH / OUTPUT_FILE
    
    # Set random seed for reproducibility
    random.seed(42)
    
    all_match_rounds = get_all_match_round_combinations()
    print(f"Found {len(all_match_rounds)} unique match-round combinations")
    
    # Assign splits to each match-round combination
    partition_data = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    
    for match_id, round_number in all_match_rounds:
        # Simple random split assignment
        split = assign_random_split()
        
        split_counts[split] += 1
        partition_data.append({
            'match_id': match_id,
            'round_number': round_number,
            'split': split
        })
    
    # Sort by split (train first, then val, then test), then by match_id and round_number
    split_order = {"train": 0, "val": 1, "test": 2}
    partition_data.sort(key=lambda x: (split_order[x['split']], x['match_id'], x['round_number']))
    
    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'split', 'match_id', 'round_number'])
        
        for index, data in enumerate(partition_data):
            writer.writerow([index, data['split'], data['match_id'], data['round_number']])
    
    if len(partition_data) > 0:
        print("Created partitioned dataset with:")
        print(f"  Train: {split_counts['train']} match-rounds ({split_counts['train']/len(partition_data)*100:.1f}%)")
        print(f"  Val: {split_counts['val']} match-rounds ({split_counts['val']/len(partition_data)*100:.1f}%)")
        print(f"  Test: {split_counts['test']} match-rounds ({split_counts['test']/len(partition_data)*100:.1f}%)")
        print(f"  Total: {len(partition_data)} match-rounds")
        print(f"Output saved to: {output_path}")
    else:
        print("No match-round combinations found. No CSV file created.")
        print(f"Please check if video files exist in: {DATA_BASE_PATH / VIDEO_DIR}")
        return


def main():
    """Main function to generate partitioned CSV file."""
    print("Creating data partition with train/val/test splits...")
    print(f"Using source base path: {SRC_BASE_PATH}")
    print(f"Using data base path: {DATA_BASE_PATH}")
    build_data_partition_csv()


if __name__ == "__main__":
    main()
