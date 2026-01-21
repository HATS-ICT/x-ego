import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tqdm import tqdm

def collect_all_coordinates(trajectory_dir):
    """
    Collect all X, Y coordinates from all CSV files in the trajectory directory.
    
    Args:
        trajectory_dir (str): Path to the trajectory directory
    
    Returns:
        tuple: (x_coords, y_coords) as numpy arrays
    """
    x_coords = []
    y_coords = []
    
    # Find all CSV files recursively
    csv_pattern = os.path.join(trajectory_dir, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    print(f"Found {len(csv_files)} CSV files to process...")
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            # Read only the X and Y columns to save memory
            df = pd.read_csv(csv_file, usecols=['X', 'Y'])
            
            # Remove any NaN values
            df = df.dropna()
            
            # Append coordinates to lists
            x_coords.extend(df['X'].tolist())
            y_coords.extend(df['Y'].tolist())
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    print(f"Collected {len(x_coords)} coordinate points")
    return np.array(x_coords), np.array(y_coords)

def normalize_coordinates(x_coords, y_coords):
    """
    Normalize coordinates to 0-1 range.
    
    Args:
        x_coords (np.array): X coordinates
        y_coords (np.array): Y coordinates
    
    Returns:
        tuple: (x_norm, y_norm) normalized coordinates
    """
    # Find min and max values
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    print(f"X range: [{x_min:.2f}, {x_max:.2f}]")
    print(f"Y range: [{y_min:.2f}, {y_max:.2f}]")
    
    # Normalize to 0-1 range
    x_norm = (x_coords - x_min) / (x_max - x_min)
    y_norm = (y_coords - y_min) / (y_max - y_min)
    
    return x_norm, y_norm

def create_heatmap_grid(x_norm, y_norm, resolution):
    """
    Create a heatmap grid by counting occurrences in each grid cell.
    
    Args:
        x_norm (np.array): Normalized X coordinates (0-1)
        y_norm (np.array): Normalized Y coordinates (0-1)
        resolution (int): Grid resolution (e.g., 10 creates 10x10 = 100 boxes)
    
    Returns:
        np.array: 2D grid with counts
    """
    # Create grid
    grid = np.zeros((resolution, resolution))
    
    # Convert normalized coordinates to grid indices
    # Ensure coordinates are within bounds [0, 1)
    x_indices = np.clip((x_norm * resolution).astype(int), 0, resolution - 1)
    y_indices = np.clip((y_norm * resolution).astype(int), 0, resolution - 1)
    
    # Count occurrences in each grid cell
    for x_idx, y_idx in zip(x_indices, y_indices):
        grid[y_idx, x_idx] += 1
    
    return grid

def plot_heatmap(grid, resolution, output_path):
    """
    Create and save a heatmap plot.
    
    Args:
        grid (np.array): 2D grid with counts
        resolution (int): Grid resolution
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with log scale for better visualization
    # Add 1 to avoid log(0)
    log_grid = np.log10(grid + 1)
    
    # Create heatmap
    sns.heatmap(log_grid, 
                annot=False, 
                cmap='viridis', 
                cbar_kws={'label': 'Log10(Count + 1)'})
    
    plt.title(f'Player Trajectory Heatmap (Resolution: {resolution}x{resolution})\nTotal Points: {int(grid.sum()):,}')
    plt.xlabel('Normalized X Coordinate')
    plt.ylabel('Normalized Y Coordinate')
    
    # Set axis labels to show normalized coordinates
    tick_positions = np.arange(0, resolution+1, max(1, resolution//5))
    tick_labels = [f'{i/resolution:.1f}' for i in tick_positions]
    
    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved: {output_path}")
    
    # Print some statistics
    max_count = grid.max()
    total_count = grid.sum()
    non_zero_cells = np.count_nonzero(grid)
    
    print(f"  - Max count in a cell: {int(max_count):,}")
    print(f"  - Total points: {int(total_count):,}")
    print(f"  - Non-zero cells: {non_zero_cells}/{resolution*resolution} ({100*non_zero_cells/(resolution*resolution):.1f}%)")

def main():
    # Configuration
    trajectory_dir = "data/trajectory"
    output_dir = "artifacts/traj_heatmap"
    resolutions = [5, 10, 20, 50, 100]  # Different resolution values to try
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting trajectory distribution analysis...")
    
    # Step 1: Collect all coordinates
    print("\n1. Collecting coordinate data...")
    x_coords, y_coords = collect_all_coordinates(trajectory_dir)
    
    if len(x_coords) == 0:
        print("No coordinate data found!")
        return
    
    # Step 2: Normalize coordinates
    print("\n2. Normalizing coordinates...")
    x_norm, y_norm = normalize_coordinates(x_coords, y_coords)
    
    # Step 3: Generate heatmaps for different resolutions
    print("\n3. Generating heatmaps...")
    for resolution in resolutions:
        print(f"\nProcessing resolution {resolution}x{resolution}...")
        
        # Create grid
        grid = create_heatmap_grid(x_norm, y_norm, resolution)
        
        # Plot and save heatmap
        output_path = os.path.join(output_dir, f"trajectory_heatmap_res{resolution}.png")
        plot_heatmap(grid, resolution, output_path)
    
    print(f"\nAll heatmaps saved to: {output_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
