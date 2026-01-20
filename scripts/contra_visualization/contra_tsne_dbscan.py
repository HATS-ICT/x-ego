"""
DBSCAN Clustering of Contrastive Embeddings with Video Examples

This script performs DBSCAN clustering on contrastive embeddings and saves
example videos for each discovered cluster.

Usage:
    python scripts/contra_visualization/contra_tsne_dbscan.py --exp_name contra_effect_check_2pov_contrastive-251002-094715-f6fs
    python scripts/contra_visualization/contra_tsne_dbscan.py --exp_name <exp_name> --eps 0.5 --min_samples 5 --num_examples_per_cluster 5
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from omegaconf import OmegaConf
import sys
import platform
from tqdm import tqdm

# Fix for cross-platform checkpoint loading
if platform.system() == 'Windows':
    import pathlib._local
    if hasattr(pathlib._local, 'PosixPath'):
        pathlib._local.PosixPath = pathlib.WindowsPath

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.cross_ego_video_location_net import CrossEgoVideoLocationNet
from data_module.enemy_location_forecast import EnemyLocationForecastDataModule
from data_module.enemy_location_nowcast import EnemyLocationNowcastDataModule
from data_module.teammate_location_forecast import TeammateLocationForecastDataModule
from utils.env_utils import get_output_base_path


def parse_args():
    parser = argparse.ArgumentParser(description='DBSCAN clustering of contrastive embeddings')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint name (if None, uses best)')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to process')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN eps parameter')
    parser.add_argument('--min_samples', type=int, default=5, help='DBSCAN min_samples parameter')
    parser.add_argument('--num_examples_per_cluster', type=int, default=5, help='Number of examples to save per cluster')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity parameter')
    return parser.parse_args()


def load_experiment_config(exp_name):
    """Load config from experiment directory."""
    output_dir = Path(get_output_base_path())
    exp_dir = output_dir / exp_name
    
    config_path = exp_dir / "hparam.yaml"
    if not config_path.exists():
        config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Config file not found in {exp_dir}")
    
    cfg = OmegaConf.load(config_path)
    
    from utils.env_utils import get_src_base_path, get_data_base_path
    cfg.path.src = str(get_src_base_path())
    cfg.path.data = str(get_data_base_path())
    cfg.path.output = str(output_dir)
    cfg.path.exp = str(exp_dir)
    cfg.path.ckpt = str(exp_dir / "checkpoint")
    
    return cfg


def find_checkpoint(checkpoint_dir, checkpoint_name=None):
    """Find best checkpoint in directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    if checkpoint_name:
        ckpt_path = checkpoint_path / checkpoint_name
        if not ckpt_path.exists():
            raise ValueError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path
    
    ckpt_files = [f for f in checkpoint_path.glob("*.ckpt") if f.name != "last.ckpt"]
    if not ckpt_files:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    best_ckpt = None
    best_loss = float('inf')
    
    for ckpt_file in ckpt_files:
        try:
            if '-l' in ckpt_file.stem:
                loss_value = float(ckpt_file.stem.split('-l')[-1])
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_ckpt = ckpt_file
        except (ValueError, IndexError):
            continue
    
    if best_ckpt is None:
        best_ckpt = ckpt_files[0]
    
    print(f"Using checkpoint: {best_ckpt.name}")
    return best_ckpt


def create_datamodule(cfg):
    """Create datamodule based on task."""
    if hasattr(cfg.data, 'task'):
        task = cfg.data.task
    else:
        labels_filename = cfg.data.labels_filename
        if 'enemy_location_nowcast' in labels_filename:
            task = 'enemy_location_nowcast'
        elif 'enemy_location_forecast' in labels_filename:
            task = 'enemy_location_forecast'
        elif 'teammate_location_forecast' in labels_filename:
            task = 'teammate_location_forecast'
        else:
            raise ValueError(f"Cannot infer task from labels_filename: {labels_filename}")
        cfg.data.task = task
    
    if task == 'enemy_location_nowcast':
        return EnemyLocationNowcastDataModule(cfg)
    elif task == 'enemy_location_forecast':
        return EnemyLocationForecastDataModule(cfg)
    elif task == 'teammate_location_forecast':
        return TeammateLocationForecastDataModule(cfg)
    else:
        raise ValueError(f"Unknown task: {task}")


def extract_embeddings_and_videos(model, dataloader, num_batches, device):
    """Extract contrastive embeddings and corresponding video tensors."""
    model.eval()
    model.to(device)
    
    embeddings_list = []
    videos_list = []
    team_sides_list = []
    
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=num_batches, desc="Extracting embeddings")
        
        for batch_idx, batch in pbar:
            if batch_idx >= num_batches:
                break
            
            video = batch['video'].to(device)
            pov_team_sides = batch['pov_team_side']
            
            B, A, T, C, H, W = video.shape
            video_reshaped = video.view(B * A, T, C, H, W)
            
            # Get embeddings through contrastive
            agent_embeddings = model.video_encoder(video_reshaped)
            agent_embeddings = model.video_projector(agent_embeddings)
            agent_embeddings = agent_embeddings.view(B, A, -1)
            
            if model.use_contrastive:
                contrastive_out = model.contrastive(agent_embeddings)
                agent_embeddings = contrastive_out['embeddings']
            
            embeddings_list.append(agent_embeddings.cpu())
            videos_list.append(video.cpu())
            
            # Record team side for each agent
            for b in range(B):
                team_side = pov_team_sides[b]
                for a in range(A):
                    team_sides_list.append(team_side)
    
    # Flatten: [N, embed_dim] and [N, T, C, H, W]
    embeddings = torch.cat(embeddings_list, dim=0).view(-1, embeddings_list[0].shape[-1]).numpy()
    videos = torch.cat(videos_list, dim=0).view(-1, T, C, H, W)
    team_sides = np.array(team_sides_list)
    
    return embeddings, videos, team_sides


def plot_tsne_with_clusters(embeddings, cluster_labels, team_sides, output_path, perplexity=30, random_state=42):
    """Plot t-SNE visualization colored by DBSCAN clusters."""
    print(f"Running t-SNE on {len(embeddings)} embeddings...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                max_iter=1000, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Color by DBSCAN cluster
    ax = axes[0]
    unique_clusters = np.unique(cluster_labels)
    
    # Plot noise points first (in gray)
    if -1 in unique_clusters:
        noise_mask = cluster_labels == -1
        ax.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1],
                  c='#CCCCCC', label='Noise', s=20, alpha=0.3, edgecolors='none')
    
    # Plot clusters with distinct colors
    cluster_ids_only = unique_clusters[unique_clusters != -1]
    n_clusters = len(cluster_ids_only)
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    for i, cluster_id in enumerate(cluster_ids_only):
        mask = cluster_labels == cluster_id
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=[colors[i]], label=f'Cluster {cluster_id}',
                  s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('DBSCAN Clusters', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Color by team (to see if clusters align with teams)
    ax = axes[1]
    team_colors = {'T': '#FF6B35', 'CT': '#004E89'}
    
    for team in np.unique(team_sides):
        mask = team_sides == team
        color = team_colors.get(team, '#808080')
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=color, label=f'Team {team}',
                  s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Team Labels', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE plot to: {output_path}")
    plt.close()


def plot_video_example(video_tensor, output_path):
    """Plot video frames from a single agent."""
    T, C, H, W = video_tensor.shape
    
    # Select 5 evenly spaced frames
    num_frames = min(5, T)
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)
    
    video = video_tensor[frame_indices].numpy()
    
    frame_list = []
    for t in range(num_frames):
        frame = video[t]  # [C, H, W]
        
        # Normalize
        fr = frame.astype(np.float32)
        if fr.max() <= 1.0 and fr.min() >= -1.0 and fr.min() < 0.0:
            fr = (fr + 1.0) / 2.0
        else:
            fmin, fmax = fr.min(), fr.max()
            if fmax > fmin:
                fr = (fr - fmin) / (fmax - fmin)
            else:
                fr = np.zeros_like(fr)
        
        fr = np.transpose(fr, (1, 2, 0))
        if fr.shape[-1] == 1:
            fr = np.repeat(fr, 3, axis=-1)
        
        frame_list.append(fr)
    
    # Concatenate frames horizontally
    combined = np.concatenate(frame_list, axis=1)
    combined = np.clip(combined, 0.0, 1.0)
    
    plt.figure(figsize=(10, 2))
    plt.imshow(combined)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("DBSCAN Clustering of Contrastive Embeddings")
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print(f"DBSCAN parameters: eps={args.eps}, min_samples={args.min_samples}")
    print()
    
    # Load config and setup
    cfg = load_experiment_config(args.exp_name)
    cfg.data.num_workers = 8
    
    datamodule = create_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup("test")
    
    # Create test dataloader with shuffling enabled
    test_dataloader = datamodule._create_dataloader(
        datamodule.test_dataset,
        shuffle=True,
        drop_last=False,
        collate_fn=datamodule._get_collate_fn()
    )
    
    checkpoint_path = find_checkpoint(cfg.path.ckpt, args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CrossEgoVideoLocationNet.load_from_checkpoint(
        str(checkpoint_path), cfg=cfg, strict=False, map_location=device
    )
    
    # Extract embeddings and videos
    print("\nExtracting embeddings and videos...")
    embeddings, videos, team_sides = extract_embeddings_and_videos(
        model, test_dataloader, args.num_batches, device
    )
    
    print(f"Extracted {len(embeddings)} embeddings")
    
    # Run DBSCAN
    print(f"\nRunning DBSCAN (eps={args.eps}, min_samples={args.min_samples})...")
    dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    cluster_labels = dbscan.fit_predict(embeddings)
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters[unique_clusters != -1])
    n_noise = np.sum(cluster_labels == -1)
    
    print(f"Found {n_clusters} clusters and {n_noise} noise points")
    
    # Create output directory
    output_dir = Path("artifacts/tsne_dbscan")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    perplexity = min(args.perplexity, len(embeddings) - 1)
    plot_tsne_with_clusters(
        embeddings, cluster_labels, team_sides,
        output_dir / "tsne_dbscan_clusters.png",
        perplexity=perplexity, random_state=42
    )
    
    # Save examples for each cluster
    print(f"\nSaving examples to {output_dir}...")
    
    for cluster_id in tqdm(unique_clusters, desc="Saving cluster examples"):
        if cluster_id == -1:
            continue  # Skip noise
        
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = len(cluster_indices)
        
        # Create cluster subfolder
        cluster_dir = output_dir / f"cluster_{cluster_id:03d}_size_{cluster_size}"
        cluster_dir.mkdir(exist_ok=True)
        
        # Save a few examples
        num_examples = min(args.num_examples_per_cluster, cluster_size)
        example_indices = np.random.choice(cluster_indices, num_examples, replace=False)
        
        for i, idx in enumerate(example_indices):
            video = videos[idx]
            team = team_sides[idx]
            output_path = cluster_dir / f"example_{i:02d}_team_{team}.png"
            plot_video_example(video, output_path)
    
    # Save summary
    summary_path = output_dir / "clustering_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("DBSCAN Clustering Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Parameters: eps={args.eps}, min_samples={args.min_samples}\n")
        f.write(f"Total samples: {len(embeddings)}\n")
        f.write(f"Number of clusters: {n_clusters}\n")
        f.write(f"Noise points: {n_noise}\n\n")
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            cluster_size = np.sum(cluster_labels == cluster_id)
            f.write(f"Cluster {cluster_id}: {cluster_size} samples\n")
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

