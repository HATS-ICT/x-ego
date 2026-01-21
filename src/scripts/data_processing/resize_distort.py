"""
Resize videos to square by distorting aspect ratio.

This script processes all videos in data/video_544x306_4fps/ and resizes them
to 306x306 (shorter side dimension), distorting the aspect ratio to make them square.
Output is saved to data/video_306x306_4fps/ while maintaining the original folder structure.

Example:
    python -m src.scripts.data_processing.resize_distort
    python -m src.scripts.data_processing.resize_distort --dry-run
    python -m src.scripts.data_processing.resize_distort --num-workers 8
"""

import argparse
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm


def get_video_dimensions(video_path: Path) -> tuple[int, int] | None:
    """
    Get video dimensions using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (width, height) or None if failed
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split(","))
        return width, height
    except Exception:
        return None


def resize_video(
    src_path: Path,
    dst_path: Path,
    target_size: int | None = None,
    dry_run: bool = False
) -> tuple[Path, bool, str]:
    """
    Resize a video to square dimensions by distorting aspect ratio.
    
    Args:
        src_path: Source video path
        dst_path: Destination video path
        target_size: Target square size (if None, uses shorter side of source)
        dry_run: If True, only print what would be done
        
    Returns:
        Tuple of (src_path, success, message)
    """
    if dry_run:
        return src_path, True, f"Would resize: {src_path} -> {dst_path}"
    
    # Create parent directory if it doesn't exist
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip if destination already exists
    if dst_path.exists():
        return src_path, True, f"Skipped (exists): {dst_path}"
    
    # Get source dimensions if target_size not specified
    if target_size is None:
        dims = get_video_dimensions(src_path)
        if dims is None:
            return src_path, False, f"Error: Could not get dimensions for {src_path}"
        width, height = dims
        target_size = min(width, height)
    
    # ffmpeg command to resize to square
    # scale=target_size:target_size forces square output (distorts aspect ratio)
    cmd = [
        "ffmpeg",
        "-i", str(src_path),
        "-vf", f"scale={target_size}:{target_size},setsar=1",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-an",
        "-y",
        str(dst_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return src_path, True, f"Resized: {src_path.name}"
    except subprocess.CalledProcessError as e:
        return src_path, False, f"Error: {e.stderr[:200]}"
    except Exception as e:
        return src_path, False, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Resize videos to square by distorting aspect ratio"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually resizing"
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
        help="Base data directory (default: from DATA_BASE_PATH env var)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Target square size in pixels (default: shorter side of source video)"
    )
    args = parser.parse_args()
    
    # Load environment variables from .env file
    project_root = Path(__file__).parent.parent.parent.parent
    load_dotenv(project_root / ".env")
    
    # Determine paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(os.environ["DATA_BASE_PATH"])
    
    src_dir = data_dir / "video_544x306_4fps"
    dst_dir = data_dir / "video_306x306_4fps"
    
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
    
    # Determine target size
    target_size = args.target_size
    if target_size is None:
        # Get dimensions from first video to determine target size
        dims = get_video_dimensions(video_files[0])
        if dims:
            target_size = min(dims)
            print(f"Using target size: {target_size}x{target_size}")
        else:
            print("Error: Could not determine video dimensions")
            return
    else:
        print(f"Using specified target size: {target_size}x{target_size}")
    
    # Create destination directory
    if not args.dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Build list of (src, dst) pairs
    resize_tasks = []
    for src_path in video_files:
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path
        resize_tasks.append((src_path, dst_path))
    
    # Process videos
    success_count = 0
    skip_count = 0
    error_count = 0
    
    print(f"\nResizing videos with {args.num_workers} workers...")
    if args.dry_run:
        print("(DRY RUN - no actual resizing)")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(resize_video, src, dst, target_size, args.dry_run): (src, dst)
            for src, dst in resize_tasks
        }
        
        with tqdm(total=len(futures), desc="Resizing") as pbar:
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
    print(f"  Resized: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(video_files)}")


if __name__ == "__main__":
    main()
