"""
Example script to inspect trajectory prediction labels and H5 data.

This demonstrates how to:
1. Load the CSV metadata
2. Load trajectories from the H5 file
3. Visualize the trajectory structure
"""

import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


def inspect_trajectory_data(csv_path: str, h5_path: str, num_samples: int = 3):
    """
    Inspect trajectory prediction data from CSV and H5 files.
    
    Args:
        csv_path: Path to the CSV metadata file
        h5_path: Path to the H5 trajectory data file
        num_samples: Number of sample segments to display
    """
    # Load CSV metadata
    print("=" * 80)
    print("Loading CSV metadata...")
    print("=" * 80)
    
    df = pd.read_csv(csv_path)
    print(f"\nTotal segments: {len(df)}")
    print(f"\nCSV columns: {list(df.columns)}")
    print("\nPartition distribution:")
    print(df['partition'].value_counts())
    
    print("\nFirst few rows:")
    print(df.head())
    
    # Load H5 trajectory data
    print("\n" + "=" * 80)
    print("Loading H5 trajectory data...")
    print("=" * 80)
    
    with h5py.File(h5_path, 'r') as f:
        trajectories = f['trajectories']
        
        print(f"\nH5 dataset shape: {trajectories.shape}")
        print(f"  Dimension 0 (segments): {trajectories.shape[0]}")
        print(f"  Dimension 1 (players): {trajectories.shape[1]}")
        print(f"  Dimension 2 (timepoints): {trajectories.shape[2]}")
        print(f"  Dimension 3 (coords): {trajectories.shape[3]}")
        
        print("\nH5 attributes:")
        for key, value in trajectories.attrs.items():
            print(f"  {key}: {value}")
        
        # Load a few sample trajectories
        print("\n" + "=" * 80)
        print(f"Sample trajectory data (first {num_samples} segments):")
        print("=" * 80)
        
        for i in range(min(num_samples, len(df))):
            row = df.iloc[i]
            h5_idx = row['h5_traj_idx']
            
            # Load trajectory from H5
            traj = trajectories[h5_idx]  # Shape: (10, 60, 2)
            
            print(f"\n--- Segment {i} ---")
            print(f"Match: {row['match_id']}, Round: {row['round_num']}")
            print(f"Partition: {row['partition']}")
            print(f"H5 index: {h5_idx}")
            print(f"Video: {row['start_seconds']:.2f}s - {row['video_end_seconds']:.2f}s")
            print(f"Trajectory: {row['start_seconds']:.2f}s - {row['trajectory_end_seconds']:.2f}s")
            print(f"Trajectory shape: {traj.shape}")
            
            print("\nPlayer information:")
            for p in range(10):
                player_id = row[f'player_{p}_id']
                player_side = row[f'player_{p}_side']
                
                # Get trajectory for this player
                player_traj = traj[p]  # Shape: (60, 2)
                
                # Show start and end positions
                start_pos = player_traj[0]
                end_pos = player_traj[-1]
                
                print(f"  Player {p} ({player_side}): {player_id}")
                print(f"    Start (t=0s):   X={start_pos[0]:.3f}, Y={start_pos[1]:.3f}")
                print(f"    End (t=15s):    X={end_pos[0]:.3f}, Y={end_pos[1]:.3f}")
                print(f"    Displacement:   ΔX={end_pos[0]-start_pos[0]:.3f}, ΔY={end_pos[1]-start_pos[1]:.3f}")


def load_trajectory_batch(csv_path: str, h5_path: str, indices: list) -> tuple:
    """
    Example of how to efficiently load a batch of trajectories.
    
    Args:
        csv_path: Path to CSV metadata
        h5_path: Path to H5 trajectory data
        indices: List of segment indices to load
        
    Returns:
        Tuple of (metadata_df, trajectories_array)
    """
    # Load metadata for selected indices
    df = pd.read_csv(csv_path)
    metadata = df.iloc[indices]
    
    # Load trajectories
    with h5py.File(h5_path, 'r') as f:
        h5_indices = metadata['h5_traj_idx'].values
        trajectories = f['trajectories'][h5_indices]  # Shape: (batch_size, 10, 60, 2)
    
    return metadata, trajectories


def preload_all_trajectories(h5_path: str) -> np.ndarray:
    """
    Preload all trajectory data into memory for fast access.
    
    This is the recommended approach for datasets that fit in memory.
    
    Args:
        h5_path: Path to H5 file
        
    Returns:
        Full trajectory array
    """
    with h5py.File(h5_path, 'r') as f:
        trajectories = f['trajectories'][:]  # Load all data into memory
    return trajectories


def plot_trajectories(csv_path: str, h5_path: str, max_segments: int = None):
    """
    Plot all trajectories as line plots with transparency.
    
    Args:
        csv_path: Path to CSV metadata
        h5_path: Path to H5 trajectory data
        max_segments: Maximum number of segments to plot (None = all)
    """
    print("\n" + "=" * 80)
    print("Plotting trajectories...")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    with h5py.File(h5_path, 'r') as f:
        trajectories = f['trajectories'][:]
    
    # Limit number of segments if specified
    if max_segments is not None:
        num_segments = min(max_segments, len(trajectories))
    else:
        num_segments = len(trajectories)
    
    print(f"Plotting {num_segments} segments with {trajectories.shape[1]} players each")
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Count trajectories by team
    ct_count = 0
    t_count = 0
    
    # Plot all trajectories
    for seg_idx in range(num_segments):
        row = df.iloc[seg_idx]
        traj = trajectories[seg_idx]  # Shape: (10, 60, 2)
        
        for player_idx in range(10):
            player_side = row[f'player_{player_idx}_side']
            player_traj = traj[player_idx]  # Shape: (60, 2)
            
            x_coords = player_traj[:, 0]
            y_coords = player_traj[:, 1]
            
            # Plot with color based on team side
            if player_side == 'ct':
                ax.plot(x_coords, y_coords, color='blue', alpha=0.01, linewidth=0.5)
                ct_count += 1
            else:  # 't'
                ax.plot(x_coords, y_coords, color='red', alpha=0.01, linewidth=0.5)
                t_count += 1
    
    # Configure plot
    ax.set_title(f'All Player Trajectories\nCT (blue, n={ct_count}) | T (red, n={t_count})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X (normalized)', fontsize=12)
    ax.set_ylabel('Y (normalized)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'15s Trajectories at 4Hz ({num_segments} segments)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    print("Displaying plot...")
    plt.show()


def main():
    """Main function to run the inspection."""
    DATA_BASE_PATH = os.getenv('DATA_BASE_PATH')
    if not DATA_BASE_PATH:
        raise ValueError("DATA_BASE_PATH environment variable not set.")
    
    # Paths to the generated files
    labels_dir = Path(DATA_BASE_PATH) / 'labels'
    csv_path = labels_dir / 'teammate_opponent_traj_prediction_5s_15s_4hz.csv'
    h5_path = labels_dir / 'teammate_opponent_traj_prediction_5s_15s_4hz.h5'
    
    # Check if files exist
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        print("Please run the labeler first to generate the data.")
        return
    
    if not h5_path.exists():
        print(f"H5 file not found: {h5_path}")
        print("Please run the labeler first to generate the data.")
        return
    
    # Inspect the data
    inspect_trajectory_data(str(csv_path), str(h5_path), num_samples=3)
    
    # Demonstrate batch loading
    print("\n" + "=" * 80)
    print("Example: Loading a batch of trajectories")
    print("=" * 80)
    
    batch_indices = [0, 1, 2]  # Load first 3 segments
    metadata, trajectories = load_trajectory_batch(str(csv_path), str(h5_path), batch_indices)
    print(f"Loaded batch with {len(batch_indices)} segments")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Trajectories shape: {trajectories.shape}")
    
    # Demonstrate preloading all data
    print("\n" + "=" * 80)
    print("Example: Preloading all trajectory data")
    print("=" * 80)
    
    all_trajectories = preload_all_trajectories(str(h5_path))
    print("Preloaded all trajectories")
    print(f"Shape: {all_trajectories.shape}")
    print(f"Memory size: {all_trajectories.nbytes / (1024**2):.2f} MB")
    print("\nThis array can now be indexed directly for fast access:")
    print("  all_trajectories[segment_idx] -> (10, 60, 2)")
    
    # Plot trajectories
    plot_trajectories(str(csv_path), str(h5_path))


if __name__ == "__main__":
    main()
