"""
Inspect enemy location nowcast label distribution.

This script analyzes the distribution of location labels in the enemy location nowcast task
to identify potential class imbalance issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


def extract_all_locations(df: pd.DataFrame) -> list:
    """
    Extract all location labels from all players in the dataset.
    
    Since POV team is randomly selected during training, we need to consider
    all players as potential enemies.
    
    Args:
        df: DataFrame containing the label data
        
    Returns:
        List of all location labels
    """
    all_locations = []
    
    # Extract locations from all 10 players
    for i in range(10):
        place_col = f'player_{i}_place'
        if place_col in df.columns:
            locations = df[place_col].tolist()
            # Filter out empty/null values
            locations = [loc for loc in locations if loc and str(loc).strip() and str(loc) != 'nan']
            all_locations.extend(locations)
    
    return all_locations


def calculate_imbalance_metrics(location_counts: Counter) -> dict:
    """
    Calculate various imbalance metrics.
    
    Args:
        location_counts: Counter object with location frequencies
        
    Returns:
        Dictionary containing imbalance metrics
    """
    if not location_counts:
        return {}
    
    counts = np.array(list(location_counts.values()))
    total_samples = counts.sum()
    
    # Class frequencies
    frequencies = counts / total_samples
    
    # Imbalance metrics
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # Coefficient of variation (CV)
    mean_count = counts.mean()
    std_count = counts.std()
    cv = std_count / mean_count if mean_count > 0 else 0
    
    # Gini coefficient (measure of inequality)
    sorted_counts = np.sort(counts)
    n = len(counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
    
    return {
        'num_classes': len(location_counts),
        'total_samples': total_samples,
        'max_count': max_count,
        'min_count': min_count,
        'mean_count': mean_count,
        'std_count': std_count,
        'imbalance_ratio': imbalance_ratio,
        'coefficient_of_variation': cv,
        'gini_coefficient': gini,
        'max_frequency': frequencies.max(),
        'min_frequency': frequencies.min()
    }


def plot_location_distribution(location_counts: Counter, title: str, save_path: Path = None):
    """
    Plot the distribution of location labels.
    
    Args:
        location_counts: Counter object with location frequencies
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    # Sort by frequency
    sorted_locations = location_counts.most_common()
    locations = [loc for loc, _ in sorted_locations]
    counts = [count for _, count in sorted_locations]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Bar plot
    bars = ax1.bar(range(len(locations)), counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Location', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'{title} - Bar Plot', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(locations)))
    ax1.set_xticklabels(locations, rotation=90, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    # Log scale plot for better visualization of imbalance
    ax2.bar(range(len(locations)), counts, color='coral', alpha=0.7)
    ax2.set_xlabel('Location', fontsize=12)
    ax2.set_ylabel('Count (log scale)', fontsize=12)
    ax2.set_title(f'{title} - Log Scale', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_xticks(range(len(locations)))
    ax2.set_xticklabels(locations, rotation=90, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_cumulative_distribution(location_counts: Counter, title: str, save_path: Path = None):
    """
    Plot cumulative distribution showing how many classes account for X% of data.
    
    Args:
        location_counts: Counter object with location frequencies
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    sorted_counts = sorted(location_counts.values(), reverse=True)
    total = sum(sorted_counts)
    cumulative_pct = np.cumsum(sorted_counts) / total * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumulative_pct) + 1), cumulative_pct, 
            marker='o', linewidth=2, markersize=6)
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50% of data')
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% of data')
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='95% of data')
    
    ax.set_xlabel('Number of Classes', fontsize=12)
    ax.set_ylabel('Cumulative Percentage of Data (%)', fontsize=12)
    ax.set_title(f'{title} - Cumulative Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def analyze_partition(df: pd.DataFrame, partition_name: str):
    """
    Analyze label distribution for a specific partition.
    
    Args:
        df: DataFrame containing the label data
        partition_name: Name of the partition (train/val/test)
    """
    print(f"\n{'='*80}")
    print(f"Analyzing partition: {partition_name.upper()}")
    print(f"{'='*80}")
    
    print(f"\nNumber of samples: {len(df)}")
    
    # Extract all locations
    all_locations = extract_all_locations(df)
    location_counts = Counter(all_locations)
    
    print(f"Total location labels: {len(all_locations)}")
    print(f"Number of unique locations: {len(location_counts)}")
    
    # Calculate imbalance metrics
    metrics = calculate_imbalance_metrics(location_counts)
    
    print(f"\n{'-'*80}")
    print("IMBALANCE METRICS")
    print(f"{'-'*80}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")
    
    # Show top locations
    print(f"\n{'-'*80}")
    print("TOP 20 MOST FREQUENT LOCATIONS")
    print(f"{'-'*80}")
    for i, (location, count) in enumerate(location_counts.most_common(20), 1):
        percentage = (count / len(all_locations)) * 100
        print(f"{i:2d}. {location:30s}: {count:6d} ({percentage:5.2f}%)")
    
    # Show bottom locations
    print(f"\n{'-'*80}")
    print("TOP 20 LEAST FREQUENT LOCATIONS")
    print(f"{'-'*80}")
    for i, (location, count) in enumerate(location_counts.most_common()[-20:][::-1], 1):
        percentage = (count / len(all_locations)) * 100
        print(f"{i:2d}. {location:30s}: {count:6d} ({percentage:5.2f}%)")
    
    return location_counts, metrics


def print_summary_statistics(all_metrics: dict):
    """
    Print summary statistics across all partitions.
    
    Args:
        all_metrics: Dictionary mapping partition names to metrics
    """
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS ACROSS PARTITIONS")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<35} {'Train':<15} {'Val':<15} {'Test':<15}")
    print('-' * 80)
    
    # Get all metric keys from first partition
    if all_metrics:
        first_partition = list(all_metrics.values())[0]
        for key in first_partition.keys():
            values = []
            for partition in ['train', 'val', 'test']:
                if partition in all_metrics:
                    value = all_metrics[partition][key]
                    if isinstance(value, float):
                        values.append(f"{value:.4f}")
                    else:
                        values.append(f"{value}")
                else:
                    values.append("N/A")
            
            print(f"{key:<35} {values[0]:<15} {values[1]:<15} {values[2]:<15}")


def main():
    """Main function to analyze enemy location nowcast labels."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    label_path = project_root / "data" / "labels" / "enemy_location_nowcast_s1s_l5s.csv"
    output_dir = project_root / "artifacts" / "label_analysis"
    
    print("="*80)
    print("ENEMY LOCATION NOWCAST - LABEL DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nReading label file: {label_path}")
    
    if not label_path.exists():
        print(f"ERROR: Label file not found at {label_path}")
        return
    
    # Load data
    df = pd.read_csv(label_path, keep_default_na=False)
    print(f"Loaded {len(df)} samples")
    
    # Check if partition column exists
    if 'partition' in df.columns:
        partitions = df['partition'].unique()
        print(f"Found partitions: {sorted(partitions)}")
    else:
        print("No partition column found. Analyzing all data together.")
        partitions = ['all']
    
    # Analyze each partition
    all_location_counts = {}
    all_metrics = {}
    
    for partition in sorted(partitions):
        if partition == 'all':
            partition_df = df
        else:
            partition_df = df[df['partition'] == partition]
        
        location_counts, metrics = analyze_partition(partition_df, partition)
        all_location_counts[partition] = location_counts
        all_metrics[partition] = metrics
        
        # Plot distribution for this partition
        plot_location_distribution(
            location_counts,
            f"Enemy Location Distribution - {partition.upper()}",
            output_dir / f"location_distribution_{partition}.png"
        )
        
        plot_cumulative_distribution(
            location_counts,
            f"Enemy Location Distribution - {partition.upper()}",
            output_dir / f"cumulative_distribution_{partition}.png"
        )
    
    # Print summary statistics
    if len(all_metrics) > 1:
        print_summary_statistics(all_metrics)
    
    # Analyze overall distribution (all partitions combined)
    if len(partitions) > 1:
        print(f"\n{'='*80}")
        print("OVERALL DISTRIBUTION (ALL PARTITIONS COMBINED)")
        print(f"{'='*80}")
        
        all_locations = extract_all_locations(df)
        overall_counts = Counter(all_locations)
        overall_metrics = calculate_imbalance_metrics(overall_counts)
        
        print(f"\nTotal samples: {len(df)}")
        print(f"Total location labels: {len(all_locations)}")
        print(f"Number of unique locations: {len(overall_counts)}")
        
        print(f"\n{'-'*80}")
        print("OVERALL IMBALANCE METRICS")
        print(f"{'-'*80}")
        for key, value in overall_metrics.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:.4f}")
            else:
                print(f"{key:30s}: {value}")
        
        # Plot overall distribution
        plot_location_distribution(
            overall_counts,
            "Enemy Location Distribution - OVERALL",
            output_dir / "location_distribution_overall.png"
        )
        
        plot_cumulative_distribution(
            overall_counts,
            "Enemy Location Distribution - OVERALL",
            output_dir / "cumulative_distribution_overall.png"
        )
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Get overall metrics for recommendations
    if len(partitions) > 1:
        metrics = overall_metrics
    else:
        metrics = all_metrics[list(all_metrics.keys())[0]]
    
    if metrics['imbalance_ratio'] > 100:
        print("\n[!] SEVERE CLASS IMBALANCE DETECTED")
        print(f"    Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
        print("\n    Recommended actions:")
        print("    1. Use class weights in loss function (inversely proportional to frequency)")
        print("    2. Consider focal loss to focus on hard/rare classes")
        print("    3. Apply oversampling for rare classes or undersampling for common classes")
        print("    4. Use stratified sampling during training")
        print("    5. Consider hierarchical/grouped classification (e.g., by map region)")
    elif metrics['imbalance_ratio'] > 10:
        print("\n[!] MODERATE CLASS IMBALANCE DETECTED")
        print(f"    Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
        print("\n    Recommended actions:")
        print("    1. Use class weights in loss function")
        print("    2. Monitor per-class metrics (not just overall accuracy)")
        print("    3. Consider focal loss or weighted sampling")
    else:
        print("\n[OK] Class distribution is relatively balanced")
        print(f"     Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Plots saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

