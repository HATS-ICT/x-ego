import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def compute_tsne(embeddings_path):
    """Compute t-SNE for embeddings from h5 file."""
    with h5py.File(embeddings_path, 'r') as f:
        embeddings = f['embeddings'][:]
        csv_indices = f['csv_indices'][:]
    
    print(f"  Computing t-SNE for {len(embeddings)} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d, csv_indices


def load_labels(csv_indices):
    """Load label data for the given CSV indices."""
    labels_path = Path("data") / "labels" / "contrastive.csv"
    df = pd.read_csv(labels_path)
    
    # Filter out -1 values (missing agents)
    valid_mask = [idx != -1 for idx in csv_indices]
    valid_indices = [idx for idx in csv_indices if idx != -1]
    
    # csv_indices are the 'idx' column values, not positional indices
    df_subset = df[df['idx'].isin(valid_indices)]
    # Sort by the order of csv_indices to maintain alignment
    df_subset = df_subset.set_index('idx').loc[valid_indices].reset_index()
    
    return df_subset, valid_mask


def plot_3x2_comparison(short_name, finetuned_2d, pretrained_2d, csv_indices, epoch=39):
    """Create 3x2 subplot comparison with different colorings."""
    df, valid_mask = load_labels(csv_indices)
    
    # Filter embeddings to only valid ones (those with matching CSV entries)
    valid_mask_array = np.array(valid_mask)
    finetuned_2d_filtered = finetuned_2d[valid_mask_array]
    pretrained_2d_filtered = pretrained_2d[valid_mask_array]
    
    print(f"  Filtered {len(finetuned_2d)} embeddings to {len(finetuned_2d_filtered)} valid ones")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    
    # Row 0: Color by pov_team_side
    team_colors = {'t': 'orange', 'ct': 'blue'}
    colors_team = df['pov_team_side'].map(team_colors)
    
    axes[0, 0].scatter(finetuned_2d_filtered[:, 0], finetuned_2d_filtered[:, 1], s=1, alpha=0.5, c=colors_team)
    axes[0, 0].set_title(f'With Contrastive (epoch {epoch})\nColored by Team Side', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    
    axes[0, 1].scatter(pretrained_2d_filtered[:, 0], pretrained_2d_filtered[:, 1], s=1, alpha=0.5, c=colors_team)
    axes[0, 1].set_title(f'Without Contrastive (pretrained)\nColored by Team Side', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    
    # Row 1: Color by start_seconds
    axes[1, 0].scatter(finetuned_2d_filtered[:, 0], finetuned_2d_filtered[:, 1], s=1, alpha=0.5, c=df['start_seconds'], cmap='viridis')
    axes[1, 0].set_title(f'With Contrastive (epoch {epoch})\nColored by Start Seconds', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    
    sc1 = axes[1, 1].scatter(pretrained_2d_filtered[:, 0], pretrained_2d_filtered[:, 1], s=1, alpha=0.5, c=df['start_seconds'], cmap='viridis')
    axes[1, 1].set_title(f'Without Contrastive (pretrained)\nColored by Start Seconds', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    plt.colorbar(sc1, ax=axes[1, 1], label='Start Seconds')
    
    # Row 2: Color by num_alive_teammates
    axes[2, 0].scatter(finetuned_2d_filtered[:, 0], finetuned_2d_filtered[:, 1], s=1, alpha=0.5, c=df['num_alive_teammates'], cmap='plasma')
    axes[2, 0].set_title(f'With Contrastive (epoch {epoch})\nColored by Num Alive Teammates', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('t-SNE 1')
    axes[2, 0].set_ylabel('t-SNE 2')
    
    sc2 = axes[2, 1].scatter(pretrained_2d_filtered[:, 0], pretrained_2d_filtered[:, 1], s=1, alpha=0.5, c=df['num_alive_teammates'], cmap='plasma')
    axes[2, 1].set_title(f'Without Contrastive (pretrained)\nColored by Num Alive Teammates', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('t-SNE 1')
    axes[2, 1].set_ylabel('t-SNE 2')
    plt.colorbar(sc2, ax=axes[2, 1], label='Num Alive Teammates')
    
    plt.suptitle(f'{short_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    artifacts_dir = Path("artifacts") / "contra_clustering"
    output_path = artifacts_dir / f"tsne_comparison_{short_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot to {output_path}")


if __name__ == "__main__":
    experiments = [
        "dinov2-ui_all",
        "dinov2-ui_minimap",
        "dinov2-ui_none",
        "siglip2-ui_all",
        "siglip2-ui_minimap",
        "siglip2-ui_none",
    ]
    
    epoch = 39
    artifacts_dir = Path("artifacts") / "contra_clustering"
    
    for short_name in experiments:
        print(f"\n{'='*80}")
        print(f"Processing: {short_name}")
        print(f"{'='*80}")
        
        finetuned_file = artifacts_dir / f"{short_name}_e{epoch}.h5"
        pretrained_file = artifacts_dir / f"{short_name}_pretrained.h5"
        
        if not finetuned_file.exists():
            print(f"  Skipping - finetuned file not found: {finetuned_file}")
            continue
        
        if not pretrained_file.exists():
            print(f"  Skipping - pretrained file not found: {pretrained_file}")
            continue
        
        # Compute t-SNE for both
        print(f"  [1/2] Computing finetuned (epoch {epoch})...")
        finetuned_2d, csv_indices = compute_tsne(finetuned_file)
        
        print(f"  [2/2] Computing pretrained...")
        pretrained_2d, _ = compute_tsne(pretrained_file)
        
        # Create 3x2 comparison plot
        print(f"  Creating 3x2 comparison plot...")
        plot_3x2_comparison(short_name, finetuned_2d, pretrained_2d, csv_indices, epoch)
    
    print(f"\n{'='*80}")
    print("All comparisons completed!")
    print(f"{'='*80}")
