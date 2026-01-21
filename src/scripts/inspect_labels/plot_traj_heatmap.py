"""
Generate heatmaps from player trajectory data with different resolutions.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Configuration
TRAJECTORY_DIR = Path("data/trajectory")
OUTPUT_DIR = Path("artifacts/traj_heatmap")
RESOLUTIONS = [100, 120, 150, 180, 200]
COLORMAP = "plasma"


def collect_trajectory_data(trajectory_dir: Path):
    """
    Recursively collect X_norm and Y_norm values from all CSV files.
    
    Args:
        trajectory_dir: Root directory containing trajectory CSV files
        
    Returns:
        Tuple of (x_norm_array, y_norm_array)
    """
    print(f"Collecting trajectory data from {trajectory_dir}...")
    
    x_norms = []
    y_norms = []
    
    csv_files = list(trajectory_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    for i, csv_file in enumerate(csv_files, 1):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(csv_files)}...")
        
        try:
            df = pd.read_csv(csv_file)
            if 'X_norm' in df.columns and 'Y_norm' in df.columns:
                x_norms.extend(df['X_norm'].values)
                y_norms.extend(df['Y_norm'].values)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    x_norms = np.array(x_norms)
    y_norms = np.array(y_norms)
    
    print(f"Collected {len(x_norms)} trajectory points")
    print(f"X_norm range: [{x_norms.min():.4f}, {x_norms.max():.4f}]")
    print(f"Y_norm range: [{y_norms.min():.4f}, {y_norms.max():.4f}]")
    
    return x_norms, y_norms


def create_heatmap(x_data, y_data, bins, output_path: Path, colormap: str = "plasma"):
    """
    Create and save a heatmap with specified number of bins.
    
    Args:
        x_data: X coordinate data (normalized)
        y_data: Y coordinate data (normalized)
        bins: Number of bins for each dimension
        output_path: Path to save the SVG file
        colormap: Matplotlib colormap name
    """
    print(f"Creating heatmap with {bins} bins...")
    
    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot heatmap with logarithmic scale for better visibility
    # Add 1 to avoid log(0)
    im = ax.imshow(
        heatmap.T,
        origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=colormap,
        aspect='auto',
        norm=LogNorm(vmin=max(1, heatmap[heatmap > 0].min()), vmax=heatmap.max()),
        interpolation='bilinear'
    )
    
    # Remove all axes, labels, ticks, and borders
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout(pad=0)
    
    # Save as SVG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved heatmap to {output_path}")


def main():
    """Main function to generate all heatmaps."""
    print("=" * 80)
    print("Player Trajectory Heatmap Generator")
    print("=" * 80)
    
    # Collect all trajectory data
    x_norms, y_norms = collect_trajectory_data(TRAJECTORY_DIR)
    
    if len(x_norms) == 0:
        print("No trajectory data found!")
        return
    
    print("\n" + "=" * 80)
    print("Generating heatmaps...")
    print("=" * 80)
    
    # Generate heatmaps with different resolutions
    for resolution in RESOLUTIONS:
        output_file = OUTPUT_DIR / f"trajectory_heatmap_{resolution}bins.svg"
        create_heatmap(x_norms, y_norms, resolution, output_file, COLORMAP)
    
    print("\n" + "=" * 80)
    print("Done! All heatmaps generated successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
