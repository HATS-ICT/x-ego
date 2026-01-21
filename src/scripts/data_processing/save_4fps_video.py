"""
Convert 30fps videos to 4fps using ffmpeg.

This script processes all videos in data/video_544x306_30fps/ and converts them
to 4fps, saving them to data/video_544x306_4fps/ while maintaining the original
folder structure.

Example:
    python scripts/data_processing/save_4fps_video.py
    python scripts/data_processing/save_4fps_video.py --dry-run
    python scripts/data_processing/save_4fps_video.py --num-workers 8
"""

import argparse
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm


def convert_video(src_path: Path, dst_path: Path, dry_run: bool = False) -> tuple[Path, bool, str]:
    """
    Convert a single video from 30fps to 4fps using ffmpeg.
    
    Args:
        src_path: Source video path
        dst_path: Destination video path
        dry_run: If True, only print what would be done
        
    Returns:
        Tuple of (src_path, success, message)
    """
    if dry_run:
        return src_path, True, f"Would convert: {src_path} -> {dst_path}"
    
    # Create parent directory if it doesn't exist
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip if destination already exists
    if dst_path.exists():
        return src_path, True, f"Skipped (exists): {dst_path}"
    
    # ffmpeg command to convert to 4fps
    # -r 4: output frame rate
    # -c:v libx264: use H.264 codec
    # -crf 18: quality (lower = better, 18 is visually lossless)
    # -preset fast: encoding speed/compression tradeoff
    # -an: no audio (game videos typically don't need audio)
    cmd = [
        "ffmpeg",
        "-i", str(src_path),
        "-r", "4",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-an",
        "-y",  # overwrite output file if exists
        str(dst_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return src_path, True, f"Converted: {src_path.name}"
    except subprocess.CalledProcessError as e:
        return src_path, False, f"Error: {e.stderr[:200]}"
    except Exception as e:
        return src_path, False, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Convert 30fps videos to 4fps using ffmpeg"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually converting"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base data directory (default: data/ relative to project root)"
    )
    args = parser.parse_args()
    
    # Load environment variables from .env file
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env")
    
    # Determine paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(os.environ["DATA_BASE_PATH"])
    
    src_dir = data_dir / "video_544x306_30fps"
    dst_dir = data_dir / "video_544x306_4fps"
    
    print(f"Source directory: {src_dir}")
    print(f"Destination directory: {dst_dir}")
    
    if not src_dir.exists():
        print(f"Error: Source directory does not exist: {src_dir}")
        return
    
    # Find all mp4 files
    video_files = list(src_dir.rglob("*.mp4"))
    print(f"Found {len(video_files)} video files")
    
    if not video_files:
        print("No video files found.")
        return
    
    # Create destination directory
    if not args.dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Build list of (src, dst) pairs
    conversion_tasks = []
    for src_path in video_files:
        # Get relative path from source directory
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path
        conversion_tasks.append((src_path, dst_path))
    
    # Process videos
    success_count = 0
    skip_count = 0
    error_count = 0
    
    print(f"\nConverting videos with {args.num_workers} workers...")
    if args.dry_run:
        print("(DRY RUN - no actual conversion)")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(convert_video, src, dst, args.dry_run): (src, dst)
            for src, dst in conversion_tasks
        }
        
        with tqdm(total=len(futures), desc="Converting") as pbar:
            for future in as_completed(futures):
                src_path, success, message = future.result()
                
                if success:
                    if "Skipped" in message:
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    error_count += 1
                    tqdm.write(f"  {message}")
                
                pbar.update(1)
    
    # Summary
    print("\nSummary:")
    print(f"  Converted: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(video_files)}")


if __name__ == "__main__":
    main()
