"""
t-SNE Combined Visualization

This script creates a 2x3 subplot grid showing embeddings before and after contrastive learning.
Top row: after contrastive, Bottom row: before contrastive
Columns: location, time, team

Usage:
    python scripts/contra_visualization/contra_tsne_combined.py --embeddings_path output/exp_name/contrastive_tsne/embeddings_b5.npz
    python scripts/contra_visualization/contra_tsne_combined.py --embeddings_path output/exp_name/contrastive_tsne/embeddings_b5_balanced_tb5_sp50.npz --perplexity 30
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Create combined t-SNE visualization from precomputed embeddings')
    parser.add_argument('--embeddings_path', type=str, required=True,
                       help='Path to precomputed embeddings npz file')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter (default: 30)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for t-SNE (default: 42)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for the plot (default: same directory as embeddings)')
    parser.add_argument('--num_time_bins', type=int, default=5,
                       help='Number of time bins for categorical display (default: 5)')
    return parser.parse_args()


def load_embeddings(load_path):
    """Load precomputed embeddings from disk."""
    print(f"Loading embeddings from: {load_path}")
    data = np.load(load_path, allow_pickle=True)
    
    embeddings_before = data['embeddings_before']
    embeddings_after = data['embeddings_after']
    team_sides = data['team_sides']
    places = data['places']
    times = data['times'] if 'times' in data else None
    
    print(f"Loaded {len(embeddings_before)} embeddings")
    print(f"  Embedding dimension: {embeddings_before.shape[1]}")
    print(f"  Teams: {len(np.unique(team_sides))} ({', '.join(np.unique(team_sides))})")
    if places is not None and places[0] is not None:
        print(f"  Locations: {len(np.unique(places))} ({', '.join(sorted(np.unique(places)))})")
    if times is not None:
        print(f"  Time range: {times.min():.2f}s - {times.max():.2f}s")
    
    return embeddings_before, embeddings_after, team_sides, places, times


def plot_combined_tsne(embeddings_before, embeddings_after, team_sides, places, times,
                       save_path, perplexity=30, random_state=42, num_time_bins=5):
    """
    Create a 2x3 subplot grid showing t-SNE visualizations.
    Top row: after contrastive (location, time, team)
    Bottom row: before contrastive (location, time, team)
    """
    print(f"\nRunning t-SNE on embeddings with perplexity={perplexity}...")
    
    # Adjust perplexity if needed
    perplexity = min(perplexity, len(embeddings_before) - 1)
    
    # Run t-SNE on both embeddings
    print("  Computing t-SNE for BEFORE embeddings...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                max_iter=1000, verbose=1)
    embeddings_2d_before = tsne.fit_transform(embeddings_before)
    
    print("  Computing t-SNE for AFTER embeddings...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                max_iter=1000, verbose=1)
    embeddings_2d_after = tsne.fit_transform(embeddings_after)
    
    # Discretize time into categorical bins
    time_bins_categorical = None
    time_bin_labels = None
    if times is not None:
        time_edges = np.linspace(times.min(), times.max(), num_time_bins + 1)
        time_bins_categorical = np.digitize(times, time_edges[1:-1])
        # Create labels for each bin
        time_bin_labels = []
        for i in range(num_time_bins):
            bin_start = time_edges[i]
            bin_end = time_edges[i + 1]
            time_bin_labels.append(f'{bin_start:.1f}-{bin_end:.1f}s')
        print(f"  Discretized time into {num_time_bins} bins: {time_bin_labels}")
    
    # Create 2x3 subplot figure with reduced height
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    
    # Prepare location colors
    if places is not None and places[0] is not None:
        unique_places = np.unique(places)
        colors_places = plt.cm.tab20(np.linspace(0, 1, len(unique_places)))
        place_to_color = {place: colors_places[i] for i, place in enumerate(unique_places)}
    
    # Prepare team colors
    team_colors = {'T': '#FF6B35', 'CT': '#004E89'}
    
    # === TOP ROW: AFTER CONTRASTIVE ===
    
    # Top-left: Location (after)
    ax = axes[0, 0]
    if places is not None and places[0] is not None:
        for place in unique_places:
            mask = places == place
            ax.scatter(embeddings_2d_after[mask, 0], embeddings_2d_after[mask, 1],
                      c=[place_to_color[place]], s=20, alpha=0.6, edgecolors='none',
                      label=place)
    ax.set_xlabel('(a) Colored by Location (After)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Top-middle: Time (after)
    ax = axes[0, 1]
    if times is not None and time_bins_categorical is not None:
        colors_time = plt.cm.viridis(np.linspace(0, 1, num_time_bins))
        for i in range(num_time_bins):
            mask = time_bins_categorical == i
            if mask.sum() > 0:
                ax.scatter(embeddings_2d_after[mask, 0], embeddings_2d_after[mask, 1],
                          c=[colors_time[i]], s=20, alpha=0.6, edgecolors='none',
                          label=time_bin_labels[i])
    ax.set_xlabel('(b) Colored by Time (After)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Top-right: Team (after)
    ax = axes[0, 2]
    unique_teams = np.unique(team_sides)
    for team in unique_teams:
        mask = team_sides == team
        color = team_colors.get(team, '#808080')
        ax.scatter(embeddings_2d_after[mask, 0], embeddings_2d_after[mask, 1],
                  c=color, s=20, alpha=0.6, edgecolors='none', label=team)
    ax.set_xlabel('(c) Colored by Team (After)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # === BOTTOM ROW: BEFORE CONTRASTIVE ===
    
    # Bottom-left: Location (before)
    ax = axes[1, 0]
    if places is not None and places[0] is not None:
        for place in unique_places:
            mask = places == place
            ax.scatter(embeddings_2d_before[mask, 0], embeddings_2d_before[mask, 1],
                      c=[place_to_color[place]], s=20, alpha=0.6, edgecolors='none',
                      label=place)
    ax.set_xlabel('(d) Colored by Location (Before)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Bottom-middle: Time (before)
    ax = axes[1, 1]
    if times is not None and time_bins_categorical is not None:
        colors_time = plt.cm.viridis(np.linspace(0, 1, num_time_bins))
        for i in range(num_time_bins):
            mask = time_bins_categorical == i
            if mask.sum() > 0:
                ax.scatter(embeddings_2d_before[mask, 0], embeddings_2d_before[mask, 1],
                          c=[colors_time[i]], s=20, alpha=0.6, edgecolors='none',
                          label=time_bin_labels[i])
    ax.set_xlabel('(e) Colored by Time (Before)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Bottom-right: Team (before)
    ax = axes[1, 2]
    for team in unique_teams:
        mask = team_sides == team
        color = team_colors.get(team, '#808080')
        ax.scatter(embeddings_2d_before[mask, 0], embeddings_2d_before[mask, 1],
                  c=color, s=20, alpha=0.6, edgecolors='none', label=team)
    ax.set_xlabel('(f) Colored by Team (Before)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    # Save as PNG
    png_path = save_path.with_suffix('.png')
    plt.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
    print(f"\nSaved PNG plot to: {png_path}")
    
    # Save as SVG
    svg_path = save_path.with_suffix('.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved SVG plot to: {svg_path}")
    
    # Save as PDF
    pdf_path = save_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved PDF plot to: {pdf_path}")
    
    plt.close()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("t-SNE Combined Visualization")
    print("=" * 80)
    
    # Load embeddings
    embeddings_path = Path(args.embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    embeddings_before, embeddings_after, team_sides, places, times = load_embeddings(embeddings_path)
    
    # Determine output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        # Save in same directory as embeddings
        output_path = embeddings_path.parent / f"tsne_combined_{embeddings_path.stem}"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create combined visualization
    plot_combined_tsne(
        embeddings_before, embeddings_after, team_sides, places, times,
        output_path,
        perplexity=args.perplexity,
        random_state=args.random_state,
        num_time_bins=args.num_time_bins
    )
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print(f"Output saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

