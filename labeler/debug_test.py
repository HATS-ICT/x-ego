#!/usr/bin/env python3
"""
Debug test script to identify why 0 segments are being found.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from labeler.teammate_location_forecast import TeammateLocationForecastCreator


def main():
    """Debug test for teammate location forecast."""
    
    # Configuration
    if sys.platform == "win32":
        DATA_DIR = r"C:\Users\wangy\projects\x-ego\data"
        OUTPUT_DIR = r"C:\Users\wangy\projects\x-ego\data\labels"
        PARTITION_CSV_PATH = r"C:\Users\wangy\projects\x-ego\data\match_round_partitioned.csv"
    else:
        DATA_DIR = "data"
        OUTPUT_DIR = "data/labels"
        PARTITION_CSV_PATH = "data/match_round_partitioned.csv"
    
    print("Debug Test: Teammate Location Forecast")
    print("=" * 50)
    
    # Create creator with debug enabled
    creator = TeammateLocationForecastCreator(
        DATA_DIR,
        OUTPUT_DIR,
        PARTITION_CSV_PATH,
        cpu_usage=0.5
    )
    
    # Enable debug mode
    creator.debug = True
    
    # Test with just a few partitions to see what happens
    config = {
        'output_file_name': 'debug_teammate_forecast.csv',
        'segment_length_sec': 5,
        'forecast_interval_sec': 10,
        'partition': ['train']  # Just train for debugging
    }
    
    # Manually test a single match-round first
    print("\n--- Testing single match-round extraction ---")
    test_match_id = "1-2b61eddf-3ab3-47ff-ac3e-3d730458667b-1-1"
    test_round_num = 1
    
    print(f"Testing match: {test_match_id}, round: {test_round_num}")
    segments = creator._extract_segments_from_round(test_match_id, test_round_num, config)
    print(f"Extracted {len(segments)} segments from test round")
    
    if len(segments) > 0:
        print("Sample segment keys:", list(segments[0].keys()))
    
    print("\n--- Testing full process ---")
    # Run the full process
    creator.process_segments(config)


if __name__ == "__main__":
    main()
