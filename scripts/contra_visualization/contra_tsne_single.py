"""
Single t-SNE visualization combining team (marker), location (color), and time (lightness).

Usage:
    python scripts/contra_visualization/contra_tsne_single.py --embeddings_path output/<exp_name>/contrastive_tsne/embeddings_b5.npz
    python scripts/contra_visualization/contra_tsne_single.py --embeddings_path <path_to_npz> --stage after --perplexity 30
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors


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


def adjust_lightness(color, amount):
    """Adjust lightness of color. amount: 0 (dark) to 1 (light)."""
    c = mcolors.to_rgb(color)
    c_hls = mcolors.rgb_to_hsv(c)
    # Adjust value (brightness) in HSV space
    c_hls[2] = 0.6 + amount * 0.4  # Range from 60% to 100% brightness
    return mcolors.hsv_to_rgb(c_hls)


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
    
    print(f"Loading embeddings from: {embeddings_path}")
    embeddings, team_sides, places, times = load_embeddings(embeddings_path, args.stage)
    
    print(f"Loaded {len(embeddings)} embeddings")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Teams: {np.unique(team_sides)}")
    print(f"  Locations: {len(np.unique(places))}")
    if times is not None:
        print(f"  Time range: {times.min():.2f}s - {times.max():.2f}s")
    
    # Adjust perplexity if needed
    perplexity = min(args.perplexity, len(embeddings) - 1)
    
    # Create output path
    output_dir = embeddings_path.parent
    save_path = output_dir / f"tsne_combined_{args.stage}"
    
    plot_combined_tsne(embeddings, team_sides, places, times, save_path, 
                       perplexity=perplexity, random_state=args.random_state)
    
    print("Done!")


if __name__ == "__main__":
    main()

