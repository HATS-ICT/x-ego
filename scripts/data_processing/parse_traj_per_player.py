#!/usr/bin/env python3
import os
import glob
import multiprocessing as mp
from pathlib import Path
from awpy import Demo
import json
import polars as pl
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# CPU configuration
CPU_COUNT =  -1 # Default -1 uses 85% of available CPUs (adjustable)
CPU_USAGE_PERCENT = 0.85  # Adjustable CPU usage percentage
OUTPUT_FULL_TRAJECTORY = False

# Get paths from environment variables
DATA_BASE_PATH = Path(os.getenv('DATA_BASE_PATH', 'data'))
if not DATA_BASE_PATH.is_absolute():
    # If relative path, make it relative to script location
    DATA_BASE_PATH = Path(__file__).resolve().parent.parent / DATA_BASE_PATH


def get_cpu_count():
    """Get the number of CPUs to use based on CPU_COUNT configuration"""
    if CPU_COUNT == -1:
        # Use percentage of available CPUs
        available_cpus = mp.cpu_count()
        return max(1, int(available_cpus * CPU_USAGE_PERCENT))
    elif CPU_COUNT > 0:
        return min(CPU_COUNT, mp.cpu_count())
    else:
        return 1


def has_required_columns(csv_file_path, required_columns):
    """Check if a CSV file has the required columns"""
    try:
        if not os.path.exists(csv_file_path):
            return False
        
        # Read just the header (first row) to check columns
        df = pd.read_csv(csv_file_path, nrows=0)
        return all(col in df.columns for col in required_columns)
    except Exception as e:
        # If there's any error reading the file, assume it needs to be processed
        print(f"Warning: Could not check columns in {csv_file_path}: {e}")
        return False


def check_demo_already_processed(output_dir, demo_id, player_alive_times, required_columns):
    """Check if a demo is already fully processed with correct columns"""
    demo_output_dir = output_dir / demo_id
    
    if not demo_output_dir.exists():
        return False
    
    # Check all expected CSV files for this demo
    for round_num, round_data in player_alive_times.items():
        for player_alive_time in round_data:
            steamid = player_alive_time["steamid"]
            csv_file = demo_output_dir / f"{steamid}" / f"round_{round_num}.csv"
            
            # If any expected file is missing or doesn't have the correct columns, reprocess
            if not has_required_columns(csv_file, required_columns):
                return False
    
    return True


def process_single_demo(args):
    """Process a single .dem file - designed for multiprocessing"""
    dem_file, output_dir, required_columns = args
    
    try:
        print(f"Checking: {os.path.basename(dem_file)}")
        
        metadata_file = dem_file.replace(".dem", ".json").replace("demo", "metadata")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        map_name = metadata["header"]["map_name"]
            
        demo_id = metadata["demo_file"].replace(".dem", "")
        player_alive_times = metadata["player_alive_times"]
        
        # Check if demo is already processed with correct columns
        if check_demo_already_processed(output_dir, demo_id, player_alive_times, required_columns):
            print(f"Skipping {os.path.basename(dem_file)} - already processed with correct columns")
            return {
                'success': True,
                'file': dem_file,
                'skipped': True
            }
        
        print(f"Processing: {os.path.basename(dem_file)}")
        output_dir = output_dir / demo_id
        os.makedirs(output_dir, exist_ok=True)
        
        # Create Demo object
        dem = Demo(dem_file, tickrate=64)
        
        # Parse with only the necessary player properties
        dem.parse(player_props=[])
        
        # Add map_name column to the DataFrame
        dem.ticks = dem.ticks.with_columns(pl.lit(map_name).alias("map_name"))
        
        for round_num, round_data in player_alive_times.items():
            for player_alive_time in round_data:
                steamid = player_alive_time["steamid"]
                start_tick = player_alive_time["alive_start_tick"]
                end_tick = player_alive_time["alive_end_tick"]-1
                
                subtable = dem.ticks.filter(
                    (pl.col("steamid").cast(pl.Int64) == int(steamid)) &
                    (pl.col("tick").cast(pl.Int64) >= int(start_tick)) &
                    (pl.col("tick").cast(pl.Int64) <= int(end_tick))
                )
                
                # Add tick_norm column (tick - start_tick)
                subtable = subtable.with_columns(
                    (pl.col("tick") - int(start_tick)).alias("tick_norm")
                )
                
                # Add game_sec column (tick_norm / 64) with 3 decimal places
                subtable = subtable.with_columns(
                    (pl.col("tick_norm") / 64.0).round(3).alias("game_sec")
                )
                
                # Select only the required columns in the specified order
                subtable = subtable.select(required_columns)
                
                os.makedirs(output_dir / f"{steamid}", exist_ok=True)
                
                output_file = output_dir / f"{steamid}" / f"round_{round_num}.csv"
                subtable.to_pandas().to_csv(output_file, index=False)
                                
        
        return {
            'success': True,
            'file': dem_file,
            'skipped': False
        }
        
    except Exception as e:
        return {
            'success': False,
            'file': dem_file,
            'error': str(e)
        }


def parse_demo_files():
    """Parse all .dem files in the demo directory and save trajectories using parallel processing"""
    
    # Define paths using environment variable
    demo_dir = DATA_BASE_PATH / "demo"
    metadata_dir = DATA_BASE_PATH / "metadata"
    output_dir = DATA_BASE_PATH / "trajectory"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .dem files
    dem_files = glob.glob(str(demo_dir / "*.dem"))
    
    if not dem_files:
        print(f"No .dem files found in {demo_dir}")
        return
    
    # Define required columns as specified
    required_columns = [
        "tick_norm", "tick", "game_sec", "round_num", "map_name",
        "steamid", "name", "side", "X", "Y", "Z", "place", "health"
    ]
    
    # Get number of CPUs to use
    num_processes = get_cpu_count()
    print(f"Found {len(dem_files)} .dem files to process using {num_processes} CPU cores")
    
    # Prepare arguments for parallel processing
    process_args = [(dem_file, output_dir, required_columns) for dem_file in dem_files]
    
    # Process files in parallel
    if num_processes > 1:
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_single_demo, process_args)
    else:
        # Fallback to sequential processing if only 1 CPU
        results = [process_single_demo(args) for args in process_args]
    
    # Print results summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    skipped = [r for r in successful if r.get('skipped', False)]
    processed = [r for r in successful if not r.get('skipped', False)]
    
    print("\nProcessing completed:")
    print(f"  Successfully processed: {len(processed)} files")
    print(f"  Skipped (already processed): {len(skipped)} files")
    print(f"  Total successful: {len(successful)} files")
    
    for result in processed:
        print(f"    -> Processed: {os.path.basename(result['file'])}")
    
    for result in skipped:
        print(f"    -> Skipped: {os.path.basename(result['file'])}")
    
    if failed:
        print(f"  Failed to process: {len(failed)} files")
        for result in failed:
            print(f"    -> {os.path.basename(result['file'])}: {result['error']}")
    
    print(f"\nUsed {num_processes} CPU cores (out of {mp.cpu_count()} available)")

def main():
    print(f"Processing demos from: {DATA_BASE_PATH / 'demo'}")
    print(f"Processing metadata from: {DATA_BASE_PATH / 'metadata'}")
    print(f"Output directory: {DATA_BASE_PATH / 'trajectory'}")
    print(f"CPU configuration: CPU_COUNT={CPU_COUNT}, CPU_USAGE_PERCENT={CPU_USAGE_PERCENT}")
    print(f"Data base path from environment: {DATA_BASE_PATH}")
    
    parse_demo_files()

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.freeze_support()
    main() 