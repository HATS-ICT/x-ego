from pathlib import Path
import subprocess
import json
from multiprocessing import Pool, cpu_count

MAPS = ["dust2", "inferno", "mirage"]

def get_video_duration(video_info):
    """Get video duration in seconds using ffprobe.
    
    Args:
        video_info: tuple of (video_path, map_name, match_id, player_id)
    
    Returns:
        tuple of (map_name, match_id, player_id, duration)
    """
    video_path, map_name, match_id, player_id = video_info
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
            return (map_name, match_id, player_id, duration)
        else:
            print(f"Warning: Could not get duration for {video_path}")
            return (map_name, match_id, player_id, 0)
    except Exception as e:
        print(f"Warning: Error processing {video_path}: {e}")
        return (map_name, match_id, player_id, 0)

def collect_video_files(data_dir):
    """Collect all video files across all maps."""
    video_info_list = []
    for map_name in MAPS:
        video_dir = Path(data_dir) / map_name / "video_544x306_30fps"
        if not video_dir.exists():
            print(f"Warning: {video_dir} does not exist, skipping.")
            continue
        for match_dir in video_dir.iterdir():
            if not match_dir.is_dir():
                continue
            for player_dir in match_dir.iterdir():
                if not player_dir.is_dir():
                    continue
                for video_file in player_dir.glob("round_*.mp4"):
                    video_info_list.append((video_file, map_name, match_dir.name, player_dir.name))
    return video_info_list

def print_map_stats(map_name, total_videos, total_duration, unique_players, unique_matches, match_durations):
    actual_rounds = total_videos // 10
    avg_match_duration = (sum(match_durations.values()) / len(match_durations) / 10) if match_durations else 0
    print(f"\n  Map: {map_name}")
    print(f"    Total Hours:         {total_duration / 3600:.2f} hours")
    print(f"    Total Videos:        {total_videos}")
    print(f"    Total Rounds:        {actual_rounds} (videos / 10 players)")
    print(f"    Unique Players:      {len(unique_players)}")
    print(f"    Unique Matches:      {len(unique_matches)}")
    print(f"    Avg Round Duration:  {total_duration / total_videos if total_videos > 0 else 0:.2f} seconds")
    print(f"    Avg Match Duration:  {avg_match_duration / 3600:.2f} hours ({avg_match_duration:.2f} seconds)")

def analyze_dataset(data_dir):
    """Analyze video dataset statistics across all maps."""
    num_processes = max(1, cpu_count() // 2)
    print(f"Using {num_processes} processes (half of {cpu_count()} CPU cores)")

    print("Collecting video files...")
    video_info_list = collect_video_files(data_dir)
    total_videos = len(video_info_list)
    print(f"Found {total_videos} video files across {len(MAPS)} maps")
    print(f"Processing videos with {num_processes} parallel workers...")

    # Per-map accumulators
    map_stats = {
        m: {
            "total_duration": 0.0,
            "total_videos": 0,
            "unique_players": set(),
            "unique_matches": set(),
            "match_durations": {},
        }
        for m in MAPS
    }
    grand_total_duration = 0.0

    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap(get_video_duration, video_info_list), 1):
            map_name, match_id, player_id, duration = result
            s = map_stats[map_name]
            s["total_duration"] += duration
            s["total_videos"] += 1
            s["unique_players"].add(player_id)
            s["unique_matches"].add(match_id)
            if match_id not in s["match_durations"]:
                s["match_durations"][match_id] = 0.0
            s["match_durations"][match_id] += duration
            grand_total_duration += duration

            if i % 100 == 0:
                print(f"Processed {i}/{total_videos} videos... ({grand_total_duration / 3600:.2f} hours so far)")

    grand_total_videos = sum(s["total_videos"] for s in map_stats.values())
    grand_unique_players = set().union(*(s["unique_players"] for s in map_stats.values()))
    grand_unique_matches = set().union(*(s["unique_matches"] for s in map_stats.values()))

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    for map_name in MAPS:
        s = map_stats[map_name]
        print_map_stats(
            map_name,
            s["total_videos"],
            s["total_duration"],
            s["unique_players"],
            s["unique_matches"],
            s["match_durations"],
        )

    print("\n" + "-" * 60)
    print("  Overall")
    print(f"    Total Hours:         {grand_total_duration / 3600:.2f} hours")
    print(f"    Total Videos:        {grand_total_videos}")
    print(f"    Total Rounds:        {grand_total_videos // 10} (videos / 10 players)")
    print(f"    Unique Players:      {len(grand_unique_players)}")
    print(f"    Unique Matches:      {len(grand_unique_matches)}")
    print("=" * 60)

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    analyze_dataset(data_dir)
