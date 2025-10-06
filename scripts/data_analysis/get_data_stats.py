from pathlib import Path
import subprocess
import json
import os
from multiprocessing import Pool, cpu_count

def get_video_duration(video_info):
    """Get video duration in seconds using ffprobe.
    
    Args:
        video_info: tuple of (video_path, match_id, player_id)
    
    Returns:
        tuple of (match_id, player_id, duration)
    """
    video_path, match_id, player_id = video_info
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return (match_id, player_id, duration)
        else:
            print(f"Warning: Could not get duration for {video_path}")
            return (match_id, player_id, 0)
    except Exception as e:
        print(f"Warning: Error processing {video_path}: {e}")
        return (match_id, player_id, 0)

def analyze_dataset(data_dir):
    """Analyze video dataset statistics."""
    video_dir = Path(data_dir) / "video_544x306_30fps"
    
    # Calculate number of processes (half of CPU cores)
    num_processes = max(1, cpu_count() // 2)
    print(f"Using {num_processes} processes (half of {cpu_count()} CPU cores)")
    
    # First, collect all video files with their metadata
    video_info_list = []
    unique_players = set()
    unique_matches = set()
    
    print("Collecting video files...")
    # Walk through directory structure: match_id/player_id/round_*.mp4
    for match_dir in video_dir.iterdir():
        if not match_dir.is_dir():
            continue
        
        match_id = match_dir.name
        unique_matches.add(match_id)
        
        for player_dir in match_dir.iterdir():
            if not player_dir.is_dir():
                continue
            
            player_id = player_dir.name
            unique_players.add(player_id)
            
            for video_file in player_dir.glob("round_*.mp4"):
                video_info_list.append((video_file, match_id, player_id))
    
    total_rounds = len(video_info_list)
    print(f"Found {total_rounds} video files")
    print(f"Processing videos with {num_processes} parallel workers...")
    
    # Process videos in parallel
    total_duration = 0.0
    match_durations = {}  # match_id -> total duration for that match
    
    with Pool(processes=num_processes) as pool:
        # Use imap for progress tracking
        results = []
        for i, result in enumerate(pool.imap(get_video_duration, video_info_list), 1):
            match_id, player_id, duration = result
            total_duration += duration
            results.append(result)
            
            # Track match durations
            if match_id not in match_durations:
                match_durations[match_id] = 0.0
            match_durations[match_id] += duration
            
            if i % 100 == 0:
                print(f"Processed {i}/{total_rounds} videos... ({total_duration / 3600:.2f} hours so far)")
    
    # Calculate statistics
    total_hours = total_duration / 3600
    actual_total_rounds = total_rounds // 10  # Each round has 10 players
    # Each match has 10 players, so divide total match duration by 10
    avg_match_duration = (sum(match_durations.values()) / len(match_durations) / 10) if match_durations else 0
    avg_match_duration_hours = avg_match_duration / 3600
    
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Total Hours:         {total_hours:.2f} hours")
    print(f"Total Duration:      {total_duration:.2f} seconds")
    print(f"Total Videos:        {total_rounds}")
    print(f"Total Rounds:        {actual_total_rounds} (videos / 10 players)")
    print(f"Unique Players:      {len(unique_players)}")
    print(f"Unique Matches:      {len(unique_matches)}")
    print(f"Avg Round Duration:  {total_duration / total_rounds if total_rounds > 0 else 0:.2f} seconds")
    print(f"Avg Match Duration:  {avg_match_duration_hours:.2f} hours ({avg_match_duration:.2f} seconds)")
    print("="*60)

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data"
    analyze_dataset(data_dir)

