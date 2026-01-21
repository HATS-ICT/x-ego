"""
Single t-SNE visualization combining team (marker), location (color), and time (lightness).
Also computes comprehensive clustering metrics (team, location, time) for both before and after embeddings.

Usage:
    python scripts/contra_visualization/contra_tsne_single.py --embeddings_path output/<exp_name>/contrastive_tsne/embeddings_b5.npz
    python scripts/contra_visualization/contra_tsne_single.py --embeddings_path <path_to_npz> --stage after --perplexity 30
    
The script will:
    1. Load both before and after embeddings from the npz file
    2. Compute clustering metrics for team, location, and time-based groupings
    3. Display and save a comprehensive comparison of metrics
    4. Generate a t-SNE visualization for the specified stage (before or after)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist


def parse_args():
    parser = argparse.ArgumentParser(description='Single t-SNE visualization')
    parser.add_argument('--embeddings_path', type=str, required=True,
                       help='Path to embeddings npz file')
    parser.add_argument('--stage', type=str, default='after', choices=['before', 'after'],
                       help='Which embeddings to visualize: before or after contrastive (default: after)')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity (default: 30)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for t-SNE (default: 42)')
    return parser.parse_args()


def load_embeddings(path, stage='after'):
    """Load embeddings from npz file."""
    data = np.load(path, allow_pickle=True)
    embeddings = data[f'embeddings_{stage}']
    team_sides = data['team_sides']
    places = data['places']
    times = data['times'] if 'times' in data else None
    return embeddings, team_sides, places, times


def load_both_embeddings(path):
    """Load both before and after embeddings from npz file."""
    data = np.load(path, allow_pickle=True)
    embeddings_before = data['embeddings_before']
    embeddings_after = data['embeddings_after']
    team_sides = data['team_sides']
    places = data['places']
    times = data['times'] if 'times' in data else None
    return embeddings_before, embeddings_after, team_sides, places, times


def adjust_lightness(color, amount):
    """Adjust lightness of color. amount: 0 (dark) to 1 (light)."""
    c = mcolors.to_rgb(color)
    c_hls = mcolors.rgb_to_hsv(c)
    # Adjust value (brightness) in HSV space
    c_hls[2] = 0.6 + amount * 0.4  # Range from 60% to 100% brightness
    return mcolors.hsv_to_rgb(c_hls)


def compute_clustering_metrics_generic(embeddings, labels, label_name="group"):
    """
    Compute clustering metrics for any categorical label.
    
    Args:
        embeddings: [N, dim] embeddings
        labels: [N] categorical labels
        label_name: Name of the label type for reporting
        
    Returns:
        Dictionary with clustering metrics
    """
    # Compute pairwise distances
    distances = cdist(embeddings, embeddings, metric='euclidean')
    
    # For each sample, compute average distance to same-label vs different-label samples
    same_label_dists = []
    diff_label_dists = []
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        label_mask = labels == label
        label_indices = np.where(label_mask)[0]
        
        for i in label_indices:
            # Same-label distances (excluding self)
            same_label_indices = [j for j in label_indices if j != i]
            if len(same_label_indices) > 0:
                same_label_dists.append(np.mean(distances[i, same_label_indices]))
            
            # Different-label distances
            diff_label_mask = ~label_mask
            diff_label_indices = np.where(diff_label_mask)[0]
            if len(diff_label_indices) > 0:
                diff_label_dists.append(np.mean(distances[i, diff_label_indices]))
    
    avg_same_label_dist = np.mean(same_label_dists) if same_label_dists else 0
    avg_diff_label_dist = np.mean(diff_label_dists) if diff_label_dists else 0
    
    # Separation ratio: higher is better (means different-label samples are farther apart)
    separation_ratio = avg_diff_label_dist / avg_same_label_dist if avg_same_label_dist > 0 else 0
    
    return {
        'label_name': label_name,
        'avg_same_dist': avg_same_label_dist,
        'avg_diff_dist': avg_diff_label_dist,
        'separation_ratio': separation_ratio,
        'num_labels': len(unique_labels)
    }


def compute_time_clustering_metrics(embeddings, times, num_bins=5):
    """
    Compute clustering metrics for time by discretizing into bins.
    
    Args:
        embeddings: [N, dim] embeddings
        times: [N] continuous time values
        num_bins: Number of bins to discretize time into
        
    Returns:
        Dictionary with clustering metrics
    """
    if times is None:
        return None
    
    # Discretize time into bins
    time_bins = np.linspace(times.min(), times.max(), num_bins + 1)
    time_labels = np.digitize(times, time_bins[1:-1])  # Bin indices 0 to num_bins-1
    
    return compute_clustering_metrics_generic(embeddings, time_labels, f"time_bin (n={num_bins})")


def compute_all_clustering_metrics(embeddings, team_sides, places, times, num_time_bins=5):
    """
    Compute clustering metrics for all available labels.
    
    Returns:
        Dictionary with metrics for each label type
    """
    metrics = {}
    
    # Team-based clustering
    metrics['team'] = compute_clustering_metrics_generic(embeddings, team_sides, "team")
    
    # Location-based clustering
    if places is not None and places[0] is not None:
        metrics['location'] = compute_clustering_metrics_generic(embeddings, places, "location")
    
    # Time-based clustering
    if times is not None:
        metrics['time'] = compute_time_clustering_metrics(embeddings, times, num_bins=num_time_bins)
    
    return metrics


def print_and_save_metrics(metrics_before, metrics_after, output_path):
    """Print and save clustering metrics comparison."""
    print("\n" + "=" * 80)
    print("CLUSTERING METRICS COMPARISON")
    print("=" * 80)
    
    # Collect all metric types
    all_metric_types = set(metrics_before.keys()) | set(metrics_after.keys())
    
    with open(output_path, 'w') as f:
        f.write("CLUSTERING METRICS COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        for metric_type in sorted(all_metric_types):
            print(f"\n{metric_type.upper()}-BASED CLUSTERING:")
            f.write(f"\n{metric_type.upper()}-BASED CLUSTERING:\n")
            f.write("-" * 80 + "\n")
            
            if metric_type in metrics_before:
                m_before = metrics_before[metric_type]
                print("  BEFORE Contrastive:")
                print(f"    Avg same-{metric_type} distance: {m_before['avg_same_dist']:.6f}")
                print(f"    Avg diff-{metric_type} distance: {m_before['avg_diff_dist']:.6f}")
                print(f"    Separation ratio: {m_before['separation_ratio']:.6f}")
                print(f"    Number of {metric_type}s: {m_before['num_labels']}")
                
                f.write("BEFORE Contrastive:\n")
                f.write(f"  Avg same-{metric_type} distance: {m_before['avg_same_dist']:.6f}\n")
                f.write(f"  Avg diff-{metric_type} distance: {m_before['avg_diff_dist']:.6f}\n")
                f.write(f"  Separation ratio: {m_before['separation_ratio']:.6f}\n")
                f.write(f"  Number of {metric_type}s: {m_before['num_labels']}\n\n")
            
            if metric_type in metrics_after:
                m_after = metrics_after[metric_type]
                print("  AFTER Contrastive:")
                print(f"    Avg same-{metric_type} distance: {m_after['avg_same_dist']:.6f}")
                print(f"    Avg diff-{metric_type} distance: {m_after['avg_diff_dist']:.6f}")
                print(f"    Separation ratio: {m_after['separation_ratio']:.6f}")
                print(f"    Number of {metric_type}s: {m_after['num_labels']}")
                
                f.write("AFTER Contrastive:\n")
                f.write(f"  Avg same-{metric_type} distance: {m_after['avg_same_dist']:.6f}\n")
                f.write(f"  Avg diff-{metric_type} distance: {m_after['avg_diff_dist']:.6f}\n")
                f.write(f"  Separation ratio: {m_after['separation_ratio']:.6f}\n")
                f.write(f"  Number of {metric_type}s: {m_after['num_labels']}\n\n")
            
            # Compute improvement
            if metric_type in metrics_before and metric_type in metrics_after:
                m_before = metrics_before[metric_type]
                m_after = metrics_after[metric_type]
                improvement = m_after['separation_ratio'] - m_before['separation_ratio']
                improvement_pct = (improvement / m_before['separation_ratio'] * 100) if m_before['separation_ratio'] > 0 else 0
                
                print(f"  Improvement: {improvement:+.6f} ({improvement_pct:+.2f}%)")
                f.write(f"Improvement: {improvement:+.6f} ({improvement_pct:+.2f}%)\n")
    
    print("=" * 80 + "\n")
    print(f"Saved metrics to: {output_path}")


def plot_combined_tsne(embeddings, team_sides, places, times, save_path, perplexity=30, random_state=42):
    """Single t-SNE plot with team (marker), location (color), time (lightness)."""
    print(f"Running t-SNE on {len(embeddings)} embeddings (perplexity={perplexity})...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                max_iter=1000, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Setup figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Get unique locations and teams
    unique_places = np.unique(places)
    unique_teams = np.unique(team_sides)
    
    # Use a beautiful colormap for locations (perceptually uniform)
    base_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_places)))
    
    # Markers for teams: circle for T, square for CT
    team_markers = {'T': 'o', 'CT': 's'}
    
    # Normalize time globally
    if times is not None:
        times_norm_global = (times - times.min()) / (times.max() - times.min())
    else:
        times_norm_global = np.zeros(len(embeddings))
    
    # Plot each location-team combination
    for place_idx, place in enumerate(unique_places):
        place_mask = places == place
        base_color = base_colors[place_idx]
        
        for team in unique_teams:
            team_mask = team_sides == team
            mask = place_mask & team_mask
            
            if mask.sum() == 0:
                continue
            
            # Get time-based lightness for this subset
            subset_times = times_norm_global[mask]
            subset_coords = embeddings_2d[mask]
            
            # Create colors with lightness gradient based on time
            colors = np.array([adjust_lightness(base_color, t) for t in subset_times])
            
            ax.scatter(subset_coords[:, 0], subset_coords[:, 1],
                      c=colors, marker=team_markers[team],
                      s=40, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Save as SVG
    svg_path = Path(save_path).with_suffix('.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"Saved to: {svg_path}")
    plt.close()


def main():
    args = parse_args()
    
    embeddings_path = Path(args.embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    print("=" * 80)
    print("Single t-SNE Visualization with Clustering Metrics")
    print("=" * 80)
    print(f"Loading embeddings from: {embeddings_path}")
    
    # Load both before and after embeddings for metrics computation
    embeddings_before, embeddings_after, team_sides, places, times = load_both_embeddings(embeddings_path)
    
    print(f"\nLoaded {len(embeddings_before)} embeddings")
    print(f"  Embedding dim: {embeddings_before.shape[1]}")
    print(f"  Teams: {np.unique(team_sides)}")
    print(f"  Locations: {len(np.unique(places))}")
    if times is not None:
        print(f"  Time range: {times.min():.2f}s - {times.max():.2f}s")
    
    # Create output directory
    output_dir = embeddings_path.parent
    
    # Compute clustering metrics for both stages
    print("\n" + "=" * 80)
    print("Computing clustering metrics for BEFORE embeddings...")
    print("=" * 80)
    metrics_before = compute_all_clustering_metrics(embeddings_before, team_sides, places, times)
    
    print("\n" + "=" * 80)
    print("Computing clustering metrics for AFTER embeddings...")
    print("=" * 80)
    metrics_after = compute_all_clustering_metrics(embeddings_after, team_sides, places, times)
    
    # Print and save metrics comparison
    metrics_path = output_dir / "clustering_metrics_comprehensive.txt"
    print_and_save_metrics(metrics_before, metrics_after, metrics_path)
    
    # Generate t-SNE visualization for the specified stage
    print(f"\nGenerating t-SNE visualization for '{args.stage}' embeddings...")
    embeddings_to_plot = embeddings_after if args.stage == 'after' else embeddings_before
    
    # Adjust perplexity if needed
    perplexity = min(args.perplexity, len(embeddings_to_plot) - 1)
    if perplexity != args.perplexity:
        print(f"Adjusted perplexity from {args.perplexity} to {perplexity}")
    
    # Create output path
    save_path = output_dir / f"tsne_combined_{args.stage}"
    
    plot_combined_tsne(embeddings_to_plot, team_sides, places, times, save_path, 
                       perplexity=perplexity, random_state=args.random_state)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

