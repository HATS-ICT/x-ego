"""
t-SNE Visualization of Contrastive Learning Effect

This script visualizes the effect of contrastive learning on agent embeddings
using t-SNE dimensionality reduction. It extracts embeddings before and after
the contrastive projection and visualizes how agents from the same batch cluster together.

Embeddings are cached to disk and automatically reused on subsequent runs for faster visualization.

Usage:
    python scripts/contra_visualization/contra_tsne.py --exp_name contra_effect_check_2pov_contrastive-251002-094715-f6fs
    python scripts/contra_visualization/contra_tsne.py --exp_name <exp_name> --checkpoint <checkpoint_name>
    python scripts/contra_visualization/contra_tsne.py --exp_name <exp_name> --num_batches 10 --perplexity 30
    python scripts/contra_visualization/contra_tsne.py --exp_name <exp_name> --recompute  # Force recomputation
    python scripts/contra_visualization/contra_tsne.py --exp_name <exp_name> --balance_data  # Balance location and time
    python scripts/contra_visualization/contra_tsne.py --exp_name <exp_name> --balance_data --num_time_bins 10 --samples_per_bin 100
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import seaborn as sns
from omegaconf import OmegaConf
import sys
import pathlib
import platform
from tqdm import tqdm

# Fix for cross-platform checkpoint loading (PosixPath on Windows)
# This is needed when loading checkpoints saved on Linux/Mac on Windows
if platform.system() == 'Windows':
    import pathlib._local
    pathlib.PosixPath = pathlib.WindowsPath
    # Also patch in the _local module for Python 3.13+
    if hasattr(pathlib._local, 'PosixPath'):
        pathlib._local.PosixPath = pathlib.WindowsPath

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.cross_ego_video_location_net import CrossEgoVideoLocationNet
from data_module.enemy_location_forecast import EnemyLocationForecastDataModule
from data_module.enemy_location_nowcast import EnemyLocationNowcastDataModule
from data_module.teammate_location_forecast import TeammateLocationForecastDataModule
from utils.config_utils import load_cfg
from utils.env_utils import get_output_base_path


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize contrastive learning effect using t-SNE')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Experiment name (e.g., contra_effect_check_2pov_contrastive-251002-094715-f6fs)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint name (if None, uses best checkpoint)')
    parser.add_argument('--num_batches', type=int, default=5,
                       help='Number of batches to visualize (default: 5)')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter (default: 30)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for t-SNE (default: 42)')
    parser.add_argument('--recompute', action='store_true',
                       help='Force recomputation of embeddings even if cached file exists')
    parser.add_argument('--balance_data', action='store_true',
                       help='Balance location and time labels by sampling a subset of test data')
    parser.add_argument('--num_time_bins', type=int, default=5,
                       help='Number of time bins for balancing (default: 5)')
    parser.add_argument('--samples_per_bin', type=int, default=50,
                       help='Target samples per location-time bin when balancing (default: 50)')
    return parser.parse_args()


def load_experiment_config(exp_name):
    """Load config from experiment directory."""
    output_dir = Path(get_output_base_path())
    exp_dir = output_dir / exp_name
    
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory not found: {exp_dir}")
    
    # Try different config file names
    config_path = exp_dir / "hparam.yaml"
    if not config_path.exists():
        config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Config file not found in {exp_dir} (looked for hparam.yaml and config.yaml)")
    
    print(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Update paths to point to the current system paths
    from utils.env_utils import get_src_base_path, get_data_base_path
    
    cfg.path.src = str(get_src_base_path())
    cfg.path.data = str(get_data_base_path())
    cfg.path.output = str(output_dir)
    cfg.path.exp = str(exp_dir)
    cfg.path.ckpt = str(exp_dir / "checkpoint")
    cfg.path.plots = str(exp_dir / "plots")
    
    return cfg


def find_checkpoint(checkpoint_dir, checkpoint_name=None):
    """Find checkpoint file in the directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    if checkpoint_name:
        # Use specified checkpoint
        ckpt_path = checkpoint_path / checkpoint_name
        if not ckpt_path.exists():
            raise ValueError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path
    
    # Find best checkpoint (lowest loss)
    ckpt_files = list(checkpoint_path.glob("*.ckpt"))
    ckpt_files = [f for f in ckpt_files if f.name != "last.ckpt"]
    
    if not ckpt_files:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    best_ckpt = None
    best_loss = float('inf')
    
    for ckpt_file in ckpt_files:
        try:
            # Extract loss from filename (format: *-l{loss}.ckpt)
            if '-l' in ckpt_file.stem:
                loss_part = ckpt_file.stem.split('-l')[-1]
                loss_value = float(loss_part)
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_ckpt = ckpt_file
        except (ValueError, IndexError):
            continue
    
    if best_ckpt is None:
        # If no loss-based checkpoint found, use the first one
        best_ckpt = ckpt_files[0]
        print(f"Warning: Could not find loss-based checkpoint, using: {best_ckpt.name}")
    else:
        print(f"Using best checkpoint: {best_ckpt.name} (val_loss: {best_loss})")
    
    return best_ckpt


def create_datamodule(cfg):
    """Create appropriate datamodule based on task or labels_filename."""
    # Try to get task from config, otherwise infer from labels_filename
    if hasattr(cfg.data, 'task'):
        task = cfg.data.task
    else:
        # Infer task from labels_filename
        labels_filename = cfg.data.labels_filename
        if 'enemy_location_nowcast' in labels_filename:
            task = 'enemy_location_nowcast'
        elif 'enemy_location_forecast' in labels_filename:
            task = 'enemy_location_forecast'
        elif 'teammate_location_forecast' in labels_filename:
            task = 'teammate_location_forecast'
        else:
            raise ValueError(f"Cannot infer task from labels_filename: {labels_filename}")
        print(f"Inferred task from labels_filename: {task}")
        # Add to config
        cfg.data.task = task
    
    if task == 'enemy_location_nowcast':
        return EnemyLocationNowcastDataModule(cfg)
    elif task == 'enemy_location_forecast':
        return EnemyLocationForecastDataModule(cfg)
    elif task == 'teammate_location_forecast':
        return TeammateLocationForecastDataModule(cfg)
    else:
        raise ValueError(f"Unknown task: {task}")


def extract_embeddings(model, dataloader, num_batches, device):
    """
    Extract embeddings before and after contrastive projection.
    
    Returns:
        embeddings_before: [N, embed_dim] embeddings before contrastive
        embeddings_after: [N, embed_dim] embeddings after contrastive
        team_sides: [N] team side for each agent ('T' or 'CT')
        places: [N] location/place for each agent
        times: [N] normalized prediction seconds for each agent (if available)
    """
    model.eval()
    model.to(device)
    
    embeddings_before_list = []
    embeddings_after_list = []
    team_sides_list = []
    places_list = []
    times_list = []
    
    with torch.no_grad():
        # Create progress bar
        pbar = tqdm(enumerate(dataloader), total=num_batches, desc="Extracting embeddings")
        
        for batch_idx, batch in pbar:
            if batch_idx >= num_batches:
                break
            
            # Move batch to device
            video = batch['video'].to(device)
            pov_team_side_encoded = batch['pov_team_side_encoded'].to(device)
            
            # Get team side strings (T or CT)
            pov_team_sides = batch['pov_team_side']  # List of 'T' or 'CT' strings
            
            # Get place information (locations)
            agent_places = batch.get('agent_places', None)  # List of lists of place strings
            
            # Get time information if available
            times = batch.get('time', None)
            
            # Encode videos
            B, A, T, C, H, W = video.shape
            video_reshaped = video.view(B * A, T, C, H, W)
            
            # Get embeddings before contrastive (after video_projector)
            agent_embeddings = model.video_encoder(video_reshaped)
            agent_embeddings = model.video_projector(agent_embeddings)
            agent_embeddings = agent_embeddings.view(B, A, -1)
            
            # Store embeddings before contrastive
            embeddings_before_list.append(agent_embeddings.cpu())
            
            # Get embeddings after contrastive (if enabled)
            if model.use_contrastive:
                contrastive_out = model.contrastive(agent_embeddings)
                agent_embeddings_after = contrastive_out['embeddings']
                embeddings_after_list.append(agent_embeddings_after.cpu())
            else:
                # If contrastive not enabled, use same as before
                embeddings_after_list.append(agent_embeddings.cpu())
            
            # Record team, places, and time
            for b in range(B):
                team_side = pov_team_sides[b]
                time_val = times[b].item() if times is not None else None
                batch_agent_places = agent_places[b] if agent_places is not None else [None] * A
                for a in range(A):
                    team_sides_list.append(team_side)
                    places_list.append(batch_agent_places[a])
                    times_list.append(time_val)
            
            # Update progress bar with current batch info
            pbar.set_postfix({'batch': f'{batch_idx + 1}/{num_batches}', 'agents': len(team_sides_list)})
    
    # Concatenate all embeddings
    embeddings_before = torch.cat(embeddings_before_list, dim=0)  # [num_batches*B, A, embed_dim]
    embeddings_after = torch.cat(embeddings_after_list, dim=0)
    
    # Flatten to [N, embed_dim] where N = num_batches * B * A
    embeddings_before = embeddings_before.view(-1, embeddings_before.shape[-1]).numpy()
    embeddings_after = embeddings_after.view(-1, embeddings_after.shape[-1]).numpy()
    
    team_sides = np.array(team_sides_list)
    places = np.array(places_list)
    times = np.array(times_list) if times_list[0] is not None else None
    
    return embeddings_before, embeddings_after, team_sides, places, times


def save_embeddings(embeddings_before, embeddings_after, team_sides, places, times, save_path):
    """Save precomputed embeddings to disk."""
    save_dict = {
        'embeddings_before': embeddings_before,
        'embeddings_after': embeddings_after,
        'team_sides': team_sides,
        'places': places,
    }
    if times is not None:
        save_dict['times'] = times
    
    np.savez_compressed(save_path, **save_dict)
    print(f"Saved embeddings to: {save_path}")


def load_embeddings(load_path):
    """Load precomputed embeddings from disk."""
    print(f"Loading cached embeddings from: {load_path}")
    data = np.load(load_path, allow_pickle=True)
    
    embeddings_before = data['embeddings_before']
    embeddings_after = data['embeddings_after']
    team_sides = data['team_sides']
    places = data['places']
    times = data['times'] if 'times' in data else None
    
    return embeddings_before, embeddings_after, team_sides, places, times


def balance_data(embeddings_before, embeddings_after, team_sides, places, times, 
                 num_time_bins=5, samples_per_bin=50, random_state=42):
    """
    Balance dataset by location and time labels.
    
    Creates stratified bins based on location × time and samples uniformly from each bin.
    
    Args:
        embeddings_before: [N, dim] embeddings before contrastive
        embeddings_after: [N, dim] embeddings after contrastive
        team_sides: [N] team side labels
        places: [N] location labels
        times: [N] time values
        num_time_bins: Number of bins to discretize time into
        samples_per_bin: Target number of samples per location-time bin
        random_state: Random seed for reproducibility
        
    Returns:
        Balanced subset of all inputs
    """
    np.random.seed(random_state)
    
    print("\n" + "=" * 80)
    print("Balancing Data by Location and Time")
    print("=" * 80)
    
    # Check if we have the necessary labels
    if places is None or places[0] is None:
        print("WARNING: No location labels available, skipping balancing")
        return embeddings_before, embeddings_after, team_sides, places, times
    
    if times is None:
        print("WARNING: No time labels available, skipping balancing")
        return embeddings_before, embeddings_after, team_sides, places, times
    
    # Print original distribution
    print(f"\nOriginal dataset: {len(embeddings_before)} samples")
    unique_places = np.unique(places)
    print(f"  Locations: {len(unique_places)} ({', '.join(sorted(unique_places))})")
    print(f"  Time range: {times.min():.2f}s - {times.max():.2f}s")
    
    # Discretize time into bins
    time_bins = np.linspace(times.min(), times.max(), num_time_bins + 1)
    time_labels = np.digitize(times, time_bins[1:-1])  # Bin indices 0 to num_time_bins-1
    
    print(f"\nTime binning: {num_time_bins} bins")
    for i in range(num_time_bins):
        bin_mask = time_labels == i
        if bin_mask.sum() > 0:
            bin_min = times[bin_mask].min()
            bin_max = times[bin_mask].max()
            print(f"  Bin {i}: [{bin_min:.2f}s, {bin_max:.2f}s] - {bin_mask.sum()} samples")
    
    # Create stratified bins: location × time
    print(f"\nCreating location × time stratification...")
    strata = {}
    for i in range(len(embeddings_before)):
        key = (places[i], time_labels[i])
        if key not in strata:
            strata[key] = []
        strata[key].append(i)
    
    print(f"  Total strata: {len(strata)}")
    
    # Print distribution of samples per stratum
    stratum_sizes = [len(indices) for indices in strata.values()]
    print(f"  Stratum size: min={min(stratum_sizes)}, max={max(stratum_sizes)}, mean={np.mean(stratum_sizes):.1f}")
    
    # Sample from each stratum
    print(f"\nSampling {samples_per_bin} samples per stratum (or all if fewer)...")
    balanced_indices = []
    
    for (place, time_bin), indices in sorted(strata.items()):
        if len(indices) <= samples_per_bin:
            # If stratum has fewer samples than target, take all
            sampled = indices
        else:
            # Otherwise, randomly sample
            sampled = np.random.choice(indices, size=samples_per_bin, replace=False).tolist()
        balanced_indices.extend(sampled)
    
    balanced_indices = np.array(balanced_indices)
    
    # Create balanced dataset
    embeddings_before_balanced = embeddings_before[balanced_indices]
    embeddings_after_balanced = embeddings_after[balanced_indices]
    team_sides_balanced = team_sides[balanced_indices]
    places_balanced = places[balanced_indices]
    times_balanced = times[balanced_indices]
    
    print(f"\nBalanced dataset: {len(balanced_indices)} samples")
    print(f"  Reduction: {len(embeddings_before) - len(balanced_indices)} samples ({(1 - len(balanced_indices)/len(embeddings_before))*100:.1f}%)")
    
    # Print balanced distribution
    print(f"\nBalanced distribution:")
    for place in sorted(unique_places):
        place_mask = places_balanced == place
        if place_mask.sum() > 0:
            place_times = times_balanced[place_mask]
            print(f"  {place}: {place_mask.sum()} samples (time: {place_times.min():.2f}s - {place_times.max():.2f}s)")
    
    print("=" * 80 + "\n")
    
    return embeddings_before_balanced, embeddings_after_balanced, team_sides_balanced, places_balanced, times_balanced


def plot_tsne_embeddings(embeddings, team_sides, places, times, title, save_path, perplexity=30, random_state=42):
    """Plot t-SNE visualization of embeddings colored by team, location, and time."""
    print(f"Running t-SNE on {len(embeddings)} embeddings with perplexity={perplexity}...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                max_iter=1000, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Determine number of subplots based on available data
    num_plots = 1  # Default: team only
    if places is not None and places[0] is not None:
        num_plots += 1  # Add place plot
    if times is not None:
        num_plots += 1  # Add time plot
    
    # Create figure with subplots in a row
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot 1: Color by location (place)
    if places is not None and places[0] is not None:
        ax = axes[plot_idx]
        unique_places = np.unique(places)
        # Use a colormap with enough distinct colors
        colors_places = plt.cm.tab20(np.linspace(0, 1, len(unique_places)))
        
        for i, place in enumerate(unique_places):
            mask = places == place
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors_places[i]], s=20, alpha=0.6, edgecolors='none')
        
        ax.axis('off')
        plot_idx += 1
    
    # Plot 2: Color by time (if available)
    if times is not None:
        ax = axes[plot_idx]
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=times, cmap='viridis', s=20, alpha=0.6, edgecolors='none')
        ax.axis('off')
        plot_idx += 1
    
    # Plot 3: Color by team
    ax = axes[plot_idx]
    unique_teams = np.unique(team_sides)
    team_colors = {'T': '#FF6B35', 'CT': '#004E89'}
    
    for team in unique_teams:
        mask = team_sides == team
        color = team_colors.get(team, '#808080')
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                  c=color, s=20, alpha=0.6, edgecolors='none')
    
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05)
    
    # Save as SVG
    svg_path = save_path.with_suffix('.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0)
    print(f"Saved SVG plot to: {svg_path}")
    plt.close()


def compute_clustering_metrics(embeddings, team_sides):
    """Compute metrics to quantify how well same-team agents cluster together."""
    from scipy.spatial.distance import cdist
    
    # Compute pairwise distances
    distances = cdist(embeddings, embeddings, metric='euclidean')
    
    # For each agent, compute average distance to same-team agents vs different-team agents
    same_team_dists = []
    diff_team_dists = []
    
    unique_teams = np.unique(team_sides)
    
    for team in unique_teams:
        team_mask = team_sides == team
        team_indices = np.where(team_mask)[0]
        
        for i in team_indices:
            # Same-team distances (excluding self)
            same_team_indices = [j for j in team_indices if j != i]
            if len(same_team_indices) > 0:
                same_team_dists.append(np.mean(distances[i, same_team_indices]))
            
            # Different-team distances
            diff_team_mask = ~team_mask
            diff_team_indices = np.where(diff_team_mask)[0]
            if len(diff_team_indices) > 0:
                diff_team_dists.append(np.mean(distances[i, diff_team_indices]))
    
    avg_same_team_dist = np.mean(same_team_dists)
    avg_diff_team_dist = np.mean(diff_team_dists)
    
    # Separation ratio: higher is better (means different-team agents are farther apart)
    separation_ratio = avg_diff_team_dist / avg_same_team_dist if avg_same_team_dist > 0 else 0
    
    return {
        'avg_same_team_dist': avg_same_team_dist,
        'avg_diff_team_dist': avg_diff_team_dist,
        'separation_ratio': separation_ratio
    }


def main():
    args = parse_args()
    
    print("=" * 80)
    print("t-SNE Visualization of Contrastive Learning Effect")
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print(f"Number of batches: {args.num_batches}")
    print(f"t-SNE perplexity: {args.perplexity}")
    if args.balance_data:
        print(f"Data balancing: ENABLED")
        print(f"  Time bins: {args.num_time_bins}")
        print(f"  Samples per bin: {args.samples_per_bin}")
    else:
        print(f"Data balancing: DISABLED")
    print()
    
    # Load config
    cfg = load_experiment_config(args.exp_name)
    cfg.data.num_workers = 10
    cfg.data.return_time = True  # Enable time information in dataset
    
    # Check if contrastive learning is enabled
    if not cfg.model.contrastive.enable:
        print("WARNING: Contrastive learning is not enabled in this experiment!")
        print("The visualization will show embeddings before and after video_projector (no change expected).")
    
    # Create datamodule first to populate config with dataset-specific info (like num_places)
    print("\nSetting up test dataloader...")
    datamodule = create_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup("test")
    
    if datamodule.test_dataset is None:
        raise ValueError("No test dataset available!")
    
    # Create test dataloader with shuffling enabled
    test_dataloader = datamodule._create_dataloader(
        datamodule.test_dataset,
        shuffle=True,
        drop_last=False,
        collate_fn=datamodule._get_collate_fn()
    )
    print(f"Test dataset size: {len(datamodule.test_dataset)}")
    
    # Create output directory
    output_dir = Path(cfg.path.exp) / "contrastive_tsne"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define embeddings cache path
    if args.balance_data:
        embeddings_cache_path = output_dir / f"embeddings_b{args.num_batches}_balanced_tb{args.num_time_bins}_sp{args.samples_per_bin}.npz"
    else:
        embeddings_cache_path = output_dir / f"embeddings_b{args.num_batches}.npz"
    
    # Check if cached embeddings exist and load them if not recomputing
    if embeddings_cache_path.exists() and not args.recompute:
        print(f"\nFound cached embeddings: {embeddings_cache_path}")
        embeddings_before, embeddings_after, team_sides, places, times = load_embeddings(embeddings_cache_path)
    else:
        if args.recompute and embeddings_cache_path.exists():
            print(f"\nRecomputing embeddings (--recompute flag set)")
        else:
            print(f"\nNo cached embeddings found, extracting from test set...")
        
        # If balancing is enabled, we need to extract from all test data
        num_batches_to_extract = len(test_dataloader) if args.balance_data else args.num_batches
        
        # Find checkpoint and load model
        checkpoint_path = find_checkpoint(cfg.path.ckpt, args.checkpoint)
        print(f"Loading checkpoint: {checkpoint_path}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = CrossEgoVideoLocationNet.load_from_checkpoint(
            str(checkpoint_path),
            cfg=cfg,
            strict=False,
            map_location=device
        )
        
        # Extract embeddings
        if args.balance_data:
            print(f"\nExtracting embeddings from ALL test set batches for balancing ({num_batches_to_extract} batches)...")
        else:
            print("\nExtracting embeddings from test set...")
        
        embeddings_before, embeddings_after, team_sides, places, times = extract_embeddings(
            model, test_dataloader, num_batches_to_extract, device
        )
        
        # Apply balancing if requested
        if args.balance_data:
            embeddings_before, embeddings_after, team_sides, places, times = balance_data(
                embeddings_before, embeddings_after, team_sides, places, times,
                num_time_bins=args.num_time_bins,
                samples_per_bin=args.samples_per_bin,
                random_state=args.random_state
            )
        
        # Save embeddings for future use
        save_embeddings(embeddings_before, embeddings_after, team_sides, places, times, embeddings_cache_path)
    
    print(f"\nExtracted {len(embeddings_before)} agent embeddings")
    print(f"Embedding dimension: {embeddings_before.shape[1]}")
    print(f"Number of teams: {len(np.unique(team_sides))} ({', '.join(np.unique(team_sides))})")
    if places is not None and places[0] is not None:
        print(f"Number of unique places: {len(np.unique(places))} ({', '.join(sorted(np.unique(places)))})")
    if times is not None:
        print(f"Time range: {times.min():.2f}s - {times.max():.2f}s")
    print(f"\nSaving results to: {output_dir}")
    
    # Compute clustering metrics
    print("\nComputing clustering metrics...")
    metrics_before = compute_clustering_metrics(embeddings_before, team_sides)
    metrics_after = compute_clustering_metrics(embeddings_after, team_sides)
    
    print("\nClustering Metrics BEFORE Contrastive:")
    print(f"  Average same-team distance: {metrics_before['avg_same_team_dist']:.4f}")
    print(f"  Average diff-team distance: {metrics_before['avg_diff_team_dist']:.4f}")
    print(f"  Separation ratio: {metrics_before['separation_ratio']:.4f}")
    
    print("\nClustering Metrics AFTER Contrastive:")
    print(f"  Average same-team distance: {metrics_after['avg_same_team_dist']:.4f}")
    print(f"  Average diff-team distance: {metrics_after['avg_diff_team_dist']:.4f}")
    print(f"  Separation ratio: {metrics_after['separation_ratio']:.4f}")
    
    improvement = metrics_after['separation_ratio'] - metrics_before['separation_ratio']
    print(f"\nImprovement in separation ratio: {improvement:.4f} ({improvement/metrics_before['separation_ratio']*100:.1f}%)")
    
    # Save metrics to file
    metrics_path = output_dir / "clustering_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("Clustering Metrics (Team-based)\n")
        f.write("=" * 50 + "\n\n")
        f.write("BEFORE Contrastive:\n")
        f.write(f"  Average same-team distance: {metrics_before['avg_same_team_dist']:.6f}\n")
        f.write(f"  Average diff-team distance: {metrics_before['avg_diff_team_dist']:.6f}\n")
        f.write(f"  Separation ratio: {metrics_before['separation_ratio']:.6f}\n\n")
        f.write("AFTER Contrastive:\n")
        f.write(f"  Average same-team distance: {metrics_after['avg_same_team_dist']:.6f}\n")
        f.write(f"  Average diff-team distance: {metrics_after['avg_diff_team_dist']:.6f}\n")
        f.write(f"  Separation ratio: {metrics_after['separation_ratio']:.6f}\n\n")
        f.write(f"Improvement: {improvement:.6f} ({improvement/metrics_before['separation_ratio']*100:.2f}%)\n")
    print(f"Saved metrics to: {metrics_path}")
    
    # Plot t-SNE visualizations
    print("\nGenerating t-SNE visualizations...")
    
    # Adjust perplexity if needed (must be less than n_samples)
    perplexity = min(args.perplexity, len(embeddings_before) - 1)
    if perplexity != args.perplexity:
        print(f"Adjusted perplexity from {args.perplexity} to {perplexity} (max for n_samples={len(embeddings_before)})")
    
    # Add suffix to filenames if using balanced data
    suffix = "_balanced" if args.balance_data else ""
    
    plot_tsne_embeddings(
        embeddings_before, team_sides, places, times,
        "Embeddings BEFORE Contrastive",
        output_dir / f"tsne_before_contrastive{suffix}",
        perplexity=perplexity,
        random_state=args.random_state
    )
    
    plot_tsne_embeddings(
        embeddings_after, team_sides, places, times,
        "Embeddings AFTER Contrastive",
        output_dir / f"tsne_after_contrastive{suffix}",
        perplexity=perplexity,
        random_state=args.random_state
    )
    
    print("\n" + "=" * 80)
    print("t-SNE Visualization Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

