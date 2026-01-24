"""
Compute min-max scaler for speed regression tasks and visualize distributions.

This script:
1. Collects all speed values from self_speed.csv and teammate_speed.csv
2. Computes a single global min-max scaler
3. Visualizes the distribution before and after normalization (without modifying data files)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def collect_all_speed_values(labels_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Collect all speed values from self_speed and teammate_speed CSVs."""
    # Load self_speed (single label column)
    self_speed_path = labels_dir / "all_tasks" / "self_speed.csv"
    self_speed_df = pd.read_csv(self_speed_path)
    self_speed_values = self_speed_df["label"].values

    # Load teammate_speed (4 label columns: label_0, label_1, label_2, label_3)
    teammate_speed_path = labels_dir / "all_tasks" / "teammate_speed.csv"
    teammate_speed_df = pd.read_csv(teammate_speed_path)
    teammate_label_cols = ["label_0", "label_1", "label_2", "label_3"]
    teammate_speed_values = teammate_speed_df[teammate_label_cols].values.flatten()

    return self_speed_values, teammate_speed_values


def compute_global_min_max(
    self_speed: np.ndarray, teammate_speed: np.ndarray
) -> tuple[float, float]:
    """Compute global min and max across all speed values."""
    all_speeds = np.concatenate([self_speed, teammate_speed])
    global_min = float(np.min(all_speeds))
    global_max = float(np.max(all_speeds))
    return global_min, global_max


def normalize_min_max(values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Apply min-max normalization to scale values to [0, 1]."""
    return (values - min_val) / (max_val - min_val)


def plot_distributions(
    self_speed: np.ndarray,
    teammate_speed: np.ndarray,
    global_min: float,
    global_max: float,
    output_dir: Path,
) -> None:
    """Plot histograms of speed distributions before and after normalization."""
    # Normalize values
    self_speed_norm = normalize_min_max(self_speed, global_min, global_max)
    teammate_speed_norm = normalize_min_max(teammate_speed, global_min, global_max)
    all_speed = np.concatenate([self_speed, teammate_speed])
    all_speed_norm = normalize_min_max(all_speed, global_min, global_max)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Before normalization
    axes[0, 0].hist(self_speed, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Self Speed (Original)")
    axes[0, 0].set_xlabel("Speed")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(global_min, color="red", linestyle="--", label=f"min={global_min:.2f}")
    axes[0, 0].axvline(global_max, color="green", linestyle="--", label=f"max={global_max:.2f}")
    axes[0, 0].legend()

    axes[0, 1].hist(teammate_speed, bins=50, color="coral", edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Teammate Speed (Original)")
    axes[0, 1].set_xlabel("Speed")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(global_min, color="red", linestyle="--", label=f"min={global_min:.2f}")
    axes[0, 1].axvline(global_max, color="green", linestyle="--", label=f"max={global_max:.2f}")
    axes[0, 1].legend()

    axes[0, 2].hist(all_speed, bins=50, color="mediumpurple", edgecolor="black", alpha=0.7)
    axes[0, 2].set_title("All Speed Combined (Original)")
    axes[0, 2].set_xlabel("Speed")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].axvline(global_min, color="red", linestyle="--", label=f"min={global_min:.2f}")
    axes[0, 2].axvline(global_max, color="green", linestyle="--", label=f"max={global_max:.2f}")
    axes[0, 2].legend()

    # Row 2: After normalization
    axes[1, 0].hist(self_speed_norm, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Self Speed (Normalized 0-1)")
    axes[1, 0].set_xlabel("Normalized Speed")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_xlim(-0.05, 1.05)

    axes[1, 1].hist(teammate_speed_norm, bins=50, color="coral", edgecolor="black", alpha=0.7)
    axes[1, 1].set_title("Teammate Speed (Normalized 0-1)")
    axes[1, 1].set_xlabel("Normalized Speed")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_xlim(-0.05, 1.05)

    axes[1, 2].hist(all_speed_norm, bins=50, color="mediumpurple", edgecolor="black", alpha=0.7)
    axes[1, 2].set_title("All Speed Combined (Normalized 0-1)")
    axes[1, 2].set_xlabel("Normalized Speed")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].set_xlim(-0.05, 1.05)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "speed_distribution_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved distribution plot to: {output_path}")
    plt.show()


def main() -> None:
    # Setup paths
    # compute_scalar.py is at: x-ego/src/scripts/scale_speed_repression/compute_scalar.py
    # parents[0] = scale_speed_repression, [1] = scripts, [2] = src, [3] = x-ego
    project_root = Path(__file__).resolve().parents[3]
    labels_dir = project_root / "data" / "labels"
    output_dir = Path(__file__).resolve().parent

    print(f"Project root: {project_root}")
    print(f"Labels dir: {labels_dir}")
    print(f"Output dir: {output_dir}")

    # Collect all speed values
    print("\nCollecting speed values...")
    self_speed, teammate_speed = collect_all_speed_values(labels_dir)

    print(f"Self speed samples: {len(self_speed)}")
    print(f"Teammate speed samples: {len(teammate_speed)}")

    # Compute global min-max
    global_min, global_max = compute_global_min_max(self_speed, teammate_speed)

    print(f"\n{'='*50}")
    print("GLOBAL MIN-MAX SCALER VALUES")
    print(f"{'='*50}")
    print(f"Global MIN: {global_min:.6f}")
    print(f"Global MAX: {global_max:.6f}")
    print(f"Range: {global_max - global_min:.6f}")
    print(f"{'='*50}")

    # Statistics before normalization
    all_speed = np.concatenate([self_speed, teammate_speed])
    print(f"\nOriginal Statistics:")
    print(f"  Mean: {np.mean(all_speed):.4f}")
    print(f"  Std:  {np.std(all_speed):.4f}")
    print(f"  Min:  {np.min(all_speed):.4f}")
    print(f"  Max:  {np.max(all_speed):.4f}")

    # Statistics after normalization
    all_speed_norm = normalize_min_max(all_speed, global_min, global_max)
    print(f"\nNormalized Statistics (0-1 range):")
    print(f"  Mean: {np.mean(all_speed_norm):.4f}")
    print(f"  Std:  {np.std(all_speed_norm):.4f}")
    print(f"  Min:  {np.min(all_speed_norm):.4f}")
    print(f"  Max:  {np.max(all_speed_norm):.4f}")

    # Plot distributions
    print("\nGenerating distribution plots...")
    plot_distributions(self_speed, teammate_speed, global_min, global_max, output_dir)


if __name__ == "__main__":
    main()
