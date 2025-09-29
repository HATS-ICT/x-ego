#!/usr/bin/env python3
"""
Production script for creating location prediction labels.
This script generates labeled datasets for enemy location nowcast,
enemy location forecast, and teammate location forecast tasks
for train, validation, and test partitions.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from labeler.enemy_location_nowcast import EnemyLocationNowcastCreator
from labeler.enemy_location_forecast import EnemyLocationForecastCreator
from labeler.teammate_location_forecast import TeammateLocationForecastCreator


def main():
    """Main function to create location prediction labels for all partitions."""
    
    # Load paths from environment variables
    DATA_BASE_PATH = os.getenv('DATA_BASE_PATH')
    if not DATA_BASE_PATH:
        raise ValueError("DATA_BASE_PATH environment variable not set. Please check your .env file.")
    
    DATA_DIR = DATA_BASE_PATH
    OUTPUT_DIR = os.path.join(DATA_BASE_PATH, 'labels')
    PARTITION_CSV_PATH = os.path.join(DATA_BASE_PATH, 'match_round_partitioned.csv')
    
    print("Creating Location Prediction Labels")
    print("=" * 50)
    print("Processing all partitions: train, val, test")
    print("CPU Usage: 90%")
    print("Stride: 1.0 seconds")
    print()
    
    # Task 1: Enemy Location Nowcast
    print("1. Creating Enemy Location Nowcast Labels...")
    print("-" * 40)
    try:
        enemy_nowcast = EnemyLocationNowcastCreator(
            DATA_DIR,
            OUTPUT_DIR,
            PARTITION_CSV_PATH,
            cpu_usage=0.9,  # High CPU usage for production
            stride_sec=1.0  # 1 second stride
        )
        
        enemy_nowcast.process_segments({
            'output_file_name': 'enemy_location_nowcast_s1s_l5s.csv',
            'segment_length_sec': 5,
            'partition': ['train', 'val', 'test']
        })
        
        print("✓ Enemy Location Nowcast labels created successfully!")
        
    except Exception as e:
        print(f"✗ Enemy Location Nowcast failed: {e}")
        return
    
    # Task 2: Enemy Location Forecast
    print("\n2. Creating Enemy Location Forecast Labels...")
    print("-" * 40)
    try:
        enemy_forecast = EnemyLocationForecastCreator(
            DATA_DIR,
            OUTPUT_DIR,
            PARTITION_CSV_PATH,
            cpu_usage=0.9,  # High CPU usage for production
            stride_sec=1.0  # 1 second stride
        )
        
        enemy_forecast.process_segments({
            'output_file_name': 'enemy_location_forecast_s1s_l5s_f10s.csv',
            'segment_length_sec': 5,
            'forecast_interval_sec': 10,
            'partition': ['train', 'val', 'test']
        })
        
        print("✓ Enemy Location Forecast labels created successfully!")
        
    except Exception as e:
        print(f"✗ Enemy Location Forecast failed: {e}")
        return
    
    # Task 3: Teammate Location Forecast
    print("\n3. Creating Teammate Location Forecast Labels...")
    print("-" * 40)
    try:
        teammate_forecast = TeammateLocationForecastCreator(
            DATA_DIR,
            OUTPUT_DIR,
            PARTITION_CSV_PATH,
            cpu_usage=0.9,  # High CPU usage for production
            stride_sec=1.0  # 1 second stride
        )
        
        teammate_forecast.process_segments({
            'output_file_name': 'teammate_location_forecast_s1s_l5s_f10s.csv',
            'segment_length_sec': 5,
            'forecast_interval_sec': 10,
            'partition': ['train', 'val', 'test']
        })
        
        print("✓ Teammate Location Forecast labels created successfully!")
        
    except Exception as e:
        print(f"✗ Teammate Location Forecast failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("All location prediction labels created successfully!")
    print("Output files saved in:", OUTPUT_DIR)
    print("- enemy_location_nowcast_s1s_l5s.csv")
    print("- enemy_location_forecast_s1s_l5s_f10s.csv") 
    print("- teammate_location_forecast_s1s_l5s_f10s.csv")


if __name__ == "__main__":
    main()
