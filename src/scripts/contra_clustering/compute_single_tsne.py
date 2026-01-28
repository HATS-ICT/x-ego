import h5py
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def compute_tsne(embeddings_path):
    """Compute t-SNE for embeddings from h5 file."""
    with h5py.File(embeddings_path, 'r') as f:
        embeddings = f['embeddings'][:]
    
    print(f"  Computing t-SNE for {len(embeddings)} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d


def plot_side_by_side(short_name, finetuned_2d, pretrained_2d, epoch=39):
    """Create side-by-side comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Finetuned (with contrastive)
    axes[0].scatter(finetuned_2d[:, 0], finetuned_2d[:, 1], s=1, alpha=0.5, c='blue')
    axes[0].set_title(f'{short_name}\nWith Contrastive (epoch {epoch})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # Pretrained (without contrastive)
    axes[1].scatter(pretrained_2d[:, 0], pretrained_2d[:, 1], s=1, alpha=0.5, c='red')
    axes[1].set_title(f'{short_name}\nWithout Contrastive (pretrained)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
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
        finetuned_2d = compute_tsne(finetuned_file)
        
        print("  [2/2] Computing pretrained...")
        pretrained_2d = compute_tsne(pretrained_file)
        
        # Create side-by-side comparison
        print("  Creating side-by-side comparison...")
        plot_side_by_side(short_name, finetuned_2d, pretrained_2d, epoch)
    
    print(f"\n{'='*80}")
    print("All comparisons completed!")
    print(f"{'='*80}")
