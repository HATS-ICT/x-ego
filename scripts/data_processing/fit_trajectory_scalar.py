"""
Fit MinMaxScaler to all trajectory CSV files and add normalized columns.

This script:
1. Recursively finds all trajectory CSV files
2. Fits a MinMaxScaler on X, Y, Z coordinates
3. Saves the scaler for later use
4. Adds X_norm, Y_norm, Z_norm columns to all CSV files
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm import tqdm


def collect_all_coordinates(trajectory_dir: Path):
    """
    Recursively collect all X, Y, Z coordinates from trajectory CSVs.
    
    Args:
        trajectory_dir: Path to trajectory directory
        
    Returns:
        numpy array of shape (n_samples, 3) with X, Y, Z coordinates
    """
    print("Collecting all coordinates from trajectory files...")
    all_coords = []
    
    # Find all CSV files recursively
    csv_files = list(trajectory_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} trajectory CSV files")
    
    for csv_file in tqdm(csv_files, desc="Collecting coordinates"):
        try:
            df = pd.read_csv(csv_file)
            # Extract X, Y, Z columns
            coords = df[['X', 'Y', 'Z']].values
            all_coords.append(coords)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    # Concatenate all coordinates
    all_coords = np.vstack(all_coords)
    print(f"Collected {len(all_coords):,} coordinate samples")
    print(f"X range: [{all_coords[:, 0].min():.2f}, {all_coords[:, 0].max():.2f}]")
    print(f"Y range: [{all_coords[:, 1].min():.2f}, {all_coords[:, 1].max():.2f}]")
    print(f"Z range: [{all_coords[:, 2].min():.2f}, {all_coords[:, 2].max():.2f}]")
    
    return all_coords, csv_files


def fit_and_save_scaler(coords, save_path: Path):
    """
    Fit MinMaxScaler on coordinates and save it.
    
    Args:
        coords: numpy array of shape (n_samples, 3)
        save_path: Path to save the scaler
        
    Returns:
        Fitted MinMaxScaler
    """
    print("\nFitting MinMaxScaler...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(coords)
    
    print("Scaler fitted:")
    print(f"  data_min_: {scaler.data_min_}")
    print(f"  data_max_: {scaler.data_max_}")
    print(f"  scale_: {scaler.scale_}")
    
    # Save scaler
    joblib.dump(scaler, save_path)
    print(f"\nScaler saved to: {save_path}")
    
    return scaler


def add_normalized_columns(csv_files, scaler):
    """
    Add X_norm, Y_norm, Z_norm columns to all CSV files.
    
    Args:
        csv_files: List of CSV file paths
        scaler: Fitted MinMaxScaler
    """
    print("\nAdding normalized columns to CSV files...")
    
    for csv_file in tqdm(csv_files, desc="Processing files"):
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Extract and normalize coordinates
            coords = df[['X', 'Y', 'Z']].values
            coords_norm = scaler.transform(coords)
            
            # Check if normalized columns already exist
            if 'X_norm' in df.columns:
                # Update existing columns
                df['X_norm'] = coords_norm[:, 0]
                df['Y_norm'] = coords_norm[:, 1]
                df['Z_norm'] = coords_norm[:, 2]
            else:
                # Find the index of Z column to insert after it
                z_idx = df.columns.get_loc('Z')
                
                # Insert normalized columns after Z
                df.insert(z_idx + 1, 'X_norm', coords_norm[:, 0])
                df.insert(z_idx + 2, 'Y_norm', coords_norm[:, 1])
                df.insert(z_idx + 3, 'Z_norm', coords_norm[:, 2])
            
            # Save back to CSV
            df.to_csv(csv_file, index=False)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    print("Done! All CSV files updated with normalized columns.")


def main():
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    trajectory_dir = project_root / "data" / "trajectory"
    scaler_save_path = project_root / "data" / "trajectory_minmax_scaler.pkl"
    
    print(f"Project root: {project_root}")
    print(f"Trajectory directory: {trajectory_dir}")
    print(f"Scaler save path: {scaler_save_path}")
    
    # Find all CSV files
    print("\nFinding all CSV files...")
    csv_files = list(trajectory_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} trajectory CSV files")
    
    # Check if scaler already exists
    if scaler_save_path.exists():
        print(f"\nScaler already exists at: {scaler_save_path}")
        print("Loading existing scaler...")
        scaler = joblib.load(scaler_save_path)
        print("Scaler loaded:")
        print(f"  data_min_: {scaler.data_min_}")
        print(f"  data_max_: {scaler.data_max_}")
        print(f"  scale_: {scaler.scale_}")
    else:
        # Step 1: Collect all coordinates
        all_coords, csv_files = collect_all_coordinates(trajectory_dir)
        
        # Step 2: Fit and save scaler
        scaler = fit_and_save_scaler(all_coords, scaler_save_path)
    
    # Step 3: Add normalized columns to all CSVs
    add_normalized_columns(csv_files, scaler)
    
    print("\n" + "="*60)
    print("All done! Summary:")
    print(f"  - Processed {len(csv_files)} CSV files")
    print(f"  - Scaler saved to: {scaler_save_path}")
    print("  - Added columns: X_norm, Y_norm, Z_norm")
    print("="*60)


if __name__ == "__main__":
    main()

