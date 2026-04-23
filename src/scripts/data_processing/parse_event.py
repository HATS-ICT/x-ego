#!/usr/bin/env python3
import os
import glob
import multiprocessing as mp
from pathlib import Path
from awpy import Demo
import json
import polars as pl
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# CPU configuration
CPU_COUNT = -1  # Default -1 uses 85% of available CPUs (adjustable)
CPU_USAGE_PERCENT = 0.85  # Adjustable CPU usage percentage

DATA_BASE_PATH = Path(os.getenv('DATA_BASE_PATH', 'data'))
if not DATA_BASE_PATH.is_absolute():
    # If relative path, make it relative to script location
    DATA_BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent / DATA_BASE_PATH


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


def process_event_dataframe(event_df, rounds_df):
    """Process event DataFrame to add tick_norm and game_sec columns and reorder columns"""
    if event_df is None or len(event_df) == 0:
        return event_df
    
    # Create a mapping of round_num to freeze_end tick
    round_freeze_map = {}
    if rounds_df is not None and len(rounds_df) > 0:
        for row in rounds_df.iter_rows(named=True):
            round_freeze_map[row['round_num']] = row['freeze_end']
    
    # Add tick_norm column (tick - freeze_end of respective round)
    def calculate_tick_norm(tick, round_num):
        freeze_end = round_freeze_map.get(round_num, 0)
        return tick - freeze_end
    
    # Apply the calculation using map_elements
    event_df = event_df.with_columns(
        pl.struct(["tick", "round_num"]).map_elements(
            lambda x: calculate_tick_norm(x["tick"], x["round_num"]),
            return_dtype=pl.Int32
        ).alias("tick_norm")
    )
    
    # Add game_sec column (tick_norm / 64) with 3 decimal places
    event_df = event_df.with_columns(
        (pl.col("tick_norm") / 64.0).round(3).alias("game_sec")
    )
    
    # Filter out rows with negative tick_norm (events before round start)
    event_df = event_df.filter(pl.col("tick_norm") >= 0)
    
    # Convert steamid columns to string to avoid scientific notation
    steamid_columns = [col for col in event_df.columns if 'steamid' in col.lower()]
    for col in steamid_columns:
        if col in event_df.columns:
            event_df = event_df.with_columns(
                pl.col(col).cast(pl.Utf8).alias(col)
            )
    
    # Remove t_side and ct_side columns
    columns_to_remove = ["t_side", "ct_side"]
    for col in columns_to_remove:
        if col in event_df.columns:
            event_df = event_df.drop(col)
    
    # Define the desired column order: tick_norm, tick, game_sec, round_num first
    priority_columns = ["tick_norm", "tick", "game_sec", "round_num"]
    
    # Get all available columns
    available_columns = event_df.columns
    
    # Create final column order: priority columns first, then remaining columns
    ordered_columns = [col for col in priority_columns if col in available_columns]
    remaining_columns = [col for col in available_columns if col not in priority_columns]
    final_column_order = ordered_columns + remaining_columns
    
    # Reorder the DataFrame
    event_df = event_df[final_column_order]
    
    return event_df


def process_single_demo(args):
    """Process a single .dem file - designed for multiprocessing"""
    dem_file, output_dir = args
    
    try:
        print(f"Processing: {os.path.basename(dem_file)}")
        
        metadata_file = dem_file.replace(".dem", ".json").replace("demo", "metadata")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        demo_id = metadata["demo_file"].replace(".dem", "")
        match_output_dir = output_dir / demo_id
        os.makedirs(match_output_dir, exist_ok=True)
        
        # Create Demo object
        dem = Demo(dem_file, tickrate=64)
        
        # Parse demo to extract all events
        dem.parse()
        
        # Extract rounds data first for tick normalization
        rounds_df = dem.rounds
        
        # Define the 5 event types and their corresponding DataFrames
        event_types = {
            'kills': dem.kills,
            'damages': dem.damages,
            'shots': dem.shots,
            'bomb': dem.bomb,
            'rounds': rounds_df,
        }
        
        # Save each event type as a CSV file
        for event_name, event_df in event_types.items():
            if event_df is not None and len(event_df) > 0:
                # For rounds, save as-is but fix bomb_plant column type
                if event_name == 'rounds':
                    processed_df = event_df
                    # Convert bomb_plant column to integer (removing .0)
                    if 'bomb_plant' in processed_df.columns:
                        processed_df = processed_df.with_columns(
                            pl.col("bomb_plant").cast(pl.Int64).alias("bomb_plant")
                        )
                else:
                    # Process other events to add tick_norm and game_sec columns relative to round freeze_end
                    processed_df = process_event_dataframe(event_df, rounds_df)
                    
                    # For kills table, remove additional unwanted columns
                    if event_name == 'kills':
                        kills_columns_to_remove = ["weapon_fauxitemid", "weapon_itemid", "weapon_originalowner_xuid", "wipe"]
                        for col in kills_columns_to_remove:
                            if col in processed_df.columns:
                                processed_df = processed_df.drop(col)
                
                output_file = match_output_dir / f"{event_name}.csv"
                # Convert to pandas and save as CSV
                pandas_df = processed_df.to_pandas()
                
                # For rounds, ensure bomb_plant is saved as integer without .0
                if event_name == 'rounds' and 'bomb_plant' in pandas_df.columns:
                    pandas_df['bomb_plant'] = pandas_df['bomb_plant'].astype('Int64')
                
                pandas_df.to_csv(output_file, index=False)
                print(f"  -> Saved {len(processed_df)} {event_name} events to {output_file}")
            else:
                print(f"  -> No {event_name} events found")
        
        return {
            'success': True,
            'file': dem_file,
            'demo_id': demo_id
        }
        
    except Exception as e:
        return {
            'success': False,
            'file': dem_file,
            'error': str(e)
        }


def parse_demo_files(map_name=None):
    """Parse all .dem files in the specified directory and save events using parallel processing"""
    
    # Define paths
    if map_name:
        demo_dir = DATA_BASE_PATH / map_name / "demo"
        output_dir = DATA_BASE_PATH / map_name / "event"
    else:
        demo_dir = DATA_BASE_PATH / "demo"
        output_dir = DATA_BASE_PATH / "event"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .dem files
    dem_files = glob.glob(str(demo_dir / "*.dem"))
    
    if not dem_files:
        print(f"No .dem files found in {demo_dir}")
        return
    
    # Get number of CPUs to use
    num_processes = get_cpu_count()
    print(f"Found {len(dem_files)} .dem files to process using {num_processes} CPU cores")
    
    # Prepare arguments for parallel processing
    process_args = [(dem_file, output_dir) for dem_file in dem_files]
    
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
    
    print("\nProcessing completed:")
    print(f"  Successfully processed: {len(successful)} files")
    
    for result in successful:
        print(f"    -> {os.path.basename(result['file'])} (Match ID: {result['demo_id']})")
    
    if failed:
        print(f"  Failed to process: {len(failed)} files")
        for result in failed:
            print(f"    -> {os.path.basename(result['file'])}: {result['error']}")
    
    print(f"\nUsed {num_processes} CPU cores (out of {mp.cpu_count()} available)")


def main():
    parser = argparse.ArgumentParser(description="Parse CS2 events")
    parser.add_argument('--map', type=str, help='Specific map name to process (e.g. dust2, inferno)')
    args = parser.parse_args()
    
    if args.map:
        demo_dir = DATA_BASE_PATH / args.map / 'demo'
        metadata_dir = DATA_BASE_PATH / args.map / 'metadata'
        event_dir = DATA_BASE_PATH / args.map / 'event'
    else:
        demo_dir = DATA_BASE_PATH / 'demo'
        metadata_dir = DATA_BASE_PATH / 'metadata'
        event_dir = DATA_BASE_PATH / 'event'

    print(f"Processing demos from: {demo_dir}")
    print(f"Processing metadata from: {metadata_dir}")
    print(f"Output directory: {event_dir}")
    print(f"CPU configuration: CPU_COUNT={CPU_COUNT}, CPU_USAGE_PERCENT={CPU_USAGE_PERCENT}")
    print(f"Data base path from environment: {DATA_BASE_PATH}")
    
    parse_demo_files(args.map)


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.freeze_support()
    main()