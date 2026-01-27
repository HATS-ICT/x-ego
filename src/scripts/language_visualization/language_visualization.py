"""
Language Visualization Script for SigLIP2 Models.

Visualizes how linguistic concepts (text-image similarities) change:
1. Before vs After: baseline (off-the-shelf pretrained) vs epoch 39, with difference matrix
2. Across training epochs: baseline, 0, 1, 2, 3, 4, 9, 14, 19, 24, 29, 34, 39

Saves visualizations to artifacts/language_visualization/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import polars as pl
from omegaconf import OmegaConf
import platform
import random
from tqdm import tqdm

from src.models.contrastive_model import ContrastiveModel
from src.dataset.dataset_utils import construct_video_path, load_video_clip
from src.utils.env_utils import get_output_base_path, get_data_base_path
from src.scripts.language_visualization.language_utils import (
    load_siglip2_model,
    get_text_embeddings,
    get_image_embeddings,
    compute_text_image_similarity,
    replace_vision_encoder_weights,
    ALL_CONCEPTS,
    CONCEPT_CATEGORIES,
    CONCEPT_TO_CATEGORY,
    CONCEPT_TO_GROUP,
    CATEGORY_GROUPS,
    GROUP_COLORS,
    get_group_color,
)

if platform.system() == 'Windows':
    import pathlib._local
    pathlib._local.PosixPath = pathlib._local.WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath


def load_checkpoint_vision_state(experiment_name: str, epoch: int):
    """
    Load vision encoder state dict from a checkpoint.
    
    Args:
        experiment_name: Name of the experiment directory
        epoch: Epoch number to load
        
    Returns:
        State dict of the vision encoder, or None if not found
    """
    output_base = Path(get_output_base_path())
    exp_dir = output_base / experiment_name
    checkpoint_dir = exp_dir / "checkpoint"
    
    ckpt_files = list(checkpoint_dir.glob(f"*-e{epoch:02d}-*.ckpt"))
    if not ckpt_files:
        print(f"  Warning: No checkpoint found for epoch {epoch}")
        return None
    
    checkpoint_path = ckpt_files[0]
    print(f"  Loading checkpoint: {checkpoint_path.name}")
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    state_dict = checkpoint['state_dict']
    state_dict = ContrastiveModel._strip_orig_mod_prefix(state_dict)
    
    # Extract vision encoder weights
    vision_state = {}
    prefix = "video_encoder.video_encoder.vision_model."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            vision_state[new_key] = v
    
    return vision_state


def load_experiment_config(experiment_name: str):
    """Load experiment config from hparam.yaml."""
    output_base = Path(get_output_base_path())
    exp_dir = output_base / experiment_name
    hparam_path = exp_dir / "hparam.yaml"
    
    cfg = OmegaConf.load(hparam_path)
    data_base = Path(get_data_base_path())
    
    path_cfg = OmegaConf.create({
        'path': {
            'exp': str(exp_dir),
            'data': str(data_base),
        },
        'data': {
            'label_path': str(data_base / cfg.data.labels_folder / cfg.data.labels_filename),
            'video_base_path': str(data_base / cfg.data.video_folder),
            'random_mask': {
                'enable': False,
            }
        }
    })
    cfg = OmegaConf.merge(cfg, path_cfg)
    return cfg


def process_video_frames(video_clip: torch.Tensor, processor, model, device) -> torch.Tensor:
    """
    Process video frames and get image embeddings.
    
    Args:
        video_clip: Raw video tensor [T, C, H, W]
        processor: SigLIP2 processor
        model: SigLIP2 model
        device: Device to use
        
    Returns:
        Image embeddings [T, embed_dim]
    """
    num_frames = video_clip.shape[0]
    
    # Process each frame through the processor
    processed_frames = []
    for i in range(num_frames):
        frame = video_clip[i]  # [C, H, W]
        # Convert to PIL-like format (HWC, uint8)
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
        
        # Process through SigLIP2 processor
        processed = processor(images=frame_np, return_tensors="pt")
        processed_frames.append(processed.pixel_values)
    
    # Stack and move to device
    pixel_values = torch.cat(processed_frames, dim=0).to(device)  # [T, C, H, W]
    
    # Get embeddings
    image_embeds = get_image_embeddings(model, pixel_values)
    
    return image_embeds


def compute_similarities_for_video(
    video_clip: torch.Tensor,
    model,
    processor,
    text_embeds: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Compute text-image similarities for all frames in a video.
    
    Args:
        video_clip: Video tensor [T, C, H, W]
        model: SigLIP2 model
        processor: SigLIP2 processor
        text_embeds: Pre-computed text embeddings [num_concepts, embed_dim]
        device: Device to use
        
    Returns:
        Similarity matrix [T, num_concepts]
    """
    # Get image embeddings for all frames
    image_embeds = process_video_frames(video_clip, processor, model, device)
    
    # Compute similarities
    similarities = compute_text_image_similarity(text_embeds, image_embeds)
    
    return similarities.cpu().numpy()


def create_before_after_plot(
    baseline_sims: np.ndarray,
    finetuned_sims: np.ndarray,
    video_clip: torch.Tensor,
    sample_idx: int,
    save_path: Path,
    selected_concepts: list = None,
):
    """
    Create a before/after visualization comparing baseline (pretrained) vs finetuned model.
    Includes a difference matrix showing what changed.
    
    Args:
        baseline_sims: Similarities from baseline (off-the-shelf pretrained) model [T, num_concepts]
        finetuned_sims: Similarities from finetuned model (epoch 39) [T, num_concepts]
        video_clip: Original video frames [T, C, H, W]
        sample_idx: Sample index for title
        save_path: Path to save the figure
        selected_concepts: Optional list of concept indices to visualize
    """
    num_frames = baseline_sims.shape[0]
    
    # Compute difference
    diff_sims = finetuned_sims - baseline_sims
    
    # Select top concepts that show the most change
    if selected_concepts is None:
        # Find concepts with largest change in mean similarity
        change = np.abs(diff_sims.mean(axis=0))
        selected_concepts = np.argsort(change)[-12:]  # Top 12 changing concepts
    
    selected_names = [ALL_CONCEPTS[i] for i in selected_concepts]
    
    # Create figure with 4 rows: frames, baseline, finetuned, difference
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 2, 2, 2], hspace=0.3)
    
    # Row 1: Sample frames
    ax_frames = fig.add_subplot(gs[0])
    frame_indices = np.linspace(0, num_frames - 1, min(8, num_frames), dtype=int)
    
    frames_to_show = []
    for idx in frame_indices:
        frame = video_clip[idx].permute(1, 2, 0).cpu().numpy()
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        frames_to_show.append(frame)
    
    combined_frames = np.concatenate(frames_to_show, axis=1)
    ax_frames.imshow(combined_frames)
    ax_frames.set_title(f"Sample {sample_idx} - Video Frames", fontsize=12, fontweight='bold')
    ax_frames.axis('off')
    
    # Row 2: Baseline (pretrained) model similarities
    ax_base = fig.add_subplot(gs[1])
    base_selected = baseline_sims[:, selected_concepts].T
    im1 = ax_base.imshow(base_selected, aspect='auto', cmap='RdYlBu_r', vmin=-0.3, vmax=0.5)
    ax_base.set_yticks(range(len(selected_names)))
    ax_base.set_yticklabels([n[:40] for n in selected_names], fontsize=8)
    ax_base.set_xlabel('Frame', fontsize=10)
    ax_base.set_title('Baseline (Off-the-shelf SigLIP2)', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax_base, label='Similarity', shrink=0.8)
    
    # Row 3: Finetuned model similarities (epoch 39)
    ax_fine = fig.add_subplot(gs[2])
    fine_selected = finetuned_sims[:, selected_concepts].T
    im2 = ax_fine.imshow(fine_selected, aspect='auto', cmap='RdYlBu_r', vmin=-0.3, vmax=0.5)
    ax_fine.set_yticks(range(len(selected_names)))
    ax_fine.set_yticklabels([n[:40] for n in selected_names], fontsize=8)
    ax_fine.set_xlabel('Frame', fontsize=10)
    ax_fine.set_title('Finetuned SigLIP2 (Epoch 39)', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax_fine, label='Similarity', shrink=0.8)
    
    # Row 4: Difference (finetuned - baseline)
    ax_diff = fig.add_subplot(gs[3])
    diff_selected = diff_sims[:, selected_concepts].T
    # Use diverging colormap centered at 0
    max_abs_diff = max(0.3, np.abs(diff_selected).max())
    im3 = ax_diff.imshow(diff_selected, aspect='auto', cmap='RdBu_r', vmin=-max_abs_diff, vmax=max_abs_diff)
    ax_diff.set_yticks(range(len(selected_names)))
    ax_diff.set_yticklabels([n[:40] for n in selected_names], fontsize=8)
    ax_diff.set_xlabel('Frame', fontsize=10)
    ax_diff.set_title('Difference (Finetuned - Baseline)', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax_diff, label='Δ Similarity', shrink=0.8)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_epoch_evolution_plot(
    epoch_similarities: dict,
    video_clip: torch.Tensor,
    sample_idx: int,
    save_path: Path,
    epochs: list,
):
    """
    Create visualization showing how concept similarities evolve over epochs.
    
    Args:
        epoch_similarities: Dict mapping epoch (or 'baseline') to similarity matrix [T, num_concepts]
        video_clip: Original video frames [T, C, H, W]
        sample_idx: Sample index for title
        save_path: Path to save the figure
        epochs: List of epochs to visualize (can include 'baseline')
    """
    num_frames = video_clip.shape[0]
    num_epochs = len(epochs)
    
    # Find concepts with interesting evolution (high variance across epochs)
    all_sims = np.stack([epoch_similarities[e] for e in epochs], axis=0)  # [epochs, T, concepts]
    mean_sims = all_sims.mean(axis=1)  # [epochs, concepts]
    variance_across_epochs = mean_sims.var(axis=0)  # [concepts]
    top_varying = np.argsort(variance_across_epochs)[-10:]  # Top 10 varying concepts
    
    selected_names = [ALL_CONCEPTS[i] for i in top_varying]
    
    # Create figure with subplots for each epoch
    fig, axes = plt.subplots(num_epochs + 1, 1, figsize=(14, 3 * (num_epochs + 1)))
    
    # First row: video frames
    frame_indices = np.linspace(0, num_frames - 1, min(10, num_frames), dtype=int)
    frames_to_show = []
    for idx in frame_indices:
        frame = video_clip[idx].permute(1, 2, 0).cpu().numpy()
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        frames_to_show.append(frame)
    
    combined_frames = np.concatenate(frames_to_show, axis=1)
    axes[0].imshow(combined_frames)
    axes[0].set_title(f"Sample {sample_idx} - Video Frames", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Subsequent rows: similarity heatmaps for each epoch
    vmin, vmax = -0.3, 0.5
    for i, epoch in enumerate(epochs):
        ax = axes[i + 1]
        sims = epoch_similarities[epoch][:, top_varying].T
        im = ax.imshow(sims, aspect='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(selected_names)))
        ax.set_yticklabels([n[:35] for n in selected_names], fontsize=7)
        ax.set_xlabel('Frame', fontsize=9)
        # Use 'Baseline' for pretrained, otherwise 'Epoch N'
        title = 'Baseline (Pretrained)' if epoch == 'baseline' else f'Epoch {epoch}'
        ax.set_title(title, fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Similarity', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_concept_trajectory_plot(
    epoch_similarities: dict,
    sample_idx: int,
    save_path: Path,
    epochs: list,
):
    """
    Create line plots showing how mean concept similarities change over epochs.
    Groups concepts by high-level categories (egocentric, teammate, enemy, global, spatial).
    
    Args:
        epoch_similarities: Dict mapping epoch (or 'baseline') to similarity matrix [T, num_concepts]
        sample_idx: Sample index for title
        save_path: Path to save the figure
        epochs: List of epochs (can include 'baseline' as first element)
    """
    # Compute mean similarity per concept per epoch
    mean_sims = {}
    for epoch in epochs:
        mean_sims[epoch] = epoch_similarities[epoch].mean(axis=0)  # [num_concepts]
    
    # Create x-axis labels and positions
    x_positions = []
    x_labels = []
    for e in epochs:
        if e == 'baseline':
            x_positions.append(-1)
            x_labels.append('Base')
        else:
            x_positions.append(e)
            x_labels.append(str(e))
    
    # Create plot for each high-level group
    group_names = list(CATEGORY_GROUPS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for group_idx, group_name in enumerate(group_names):
        if group_idx >= len(axes):
            break
        ax = axes[group_idx]
        group_color = get_group_color(group_name)
        
        # Get all concepts in this group
        group_concepts = []
        for category in CATEGORY_GROUPS[group_name]:
            group_concepts.extend(CONCEPT_CATEGORIES.get(category, []))
        
        # Sample concepts if too many (for readability)
        if len(group_concepts) > 15:
            # Select concepts with highest variance across epochs
            variances = []
            for concept in group_concepts:
                concept_idx = ALL_CONCEPTS.index(concept)
                values = [mean_sims[e][concept_idx] for e in epochs]
                variances.append(np.var(values))
            sorted_indices = np.argsort(variances)[-15:]
            group_concepts = [group_concepts[i] for i in sorted_indices]
        
        # Use different shades of the group color
        n_concepts = len(group_concepts)
        colors = plt.colormaps['tab20'](np.linspace(0, 1, min(20, n_concepts)))
        
        for i, concept in enumerate(group_concepts):
            concept_idx = ALL_CONCEPTS.index(concept)
            values = [mean_sims[e][concept_idx] for e in epochs]
            ax.plot(x_positions, values, marker='o', label=concept[:25], 
                    color=colors[i % len(colors)], markersize=3, linewidth=1.2, alpha=0.8)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45)
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Mean Similarity', fontsize=9)
        ax.set_title(f'{group_name.title()} ({len(group_concepts)} concepts)', 
                     fontsize=11, fontweight='bold', color=group_color)
        ax.legend(fontsize=5, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
    
    # Hide extra axes
    for idx in range(len(group_names), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Sample {sample_idx} - Concept Evolution Over Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_ranking_change_plot(
    epoch_similarities: dict,
    sample_idx: int,
    save_path: Path,
    epochs: list,
):
    """
    Create a bump chart showing how concept rankings change over epochs.
    Split into two subplots:
    - Left: Early training (baseline, 0, 1, 2, 3, 4)
    - Right: Full training (baseline, 4, 9, 14, 19, 24, 29, 34, 39)
    
    Args:
        epoch_similarities: Dict mapping epoch (or 'baseline') to similarity matrix [T, num_concepts]
        sample_idx: Sample index for title
        save_path: Path to save the figure
        epochs: List of epochs (can include 'baseline' as first element)
    """
    from matplotlib.lines import Line2D
    
    # Define the two epoch splits
    early_epochs = ['baseline', 0, 1, 2, 3, 4]
    full_epochs = ['baseline', 4, 9, 14, 19, 24, 29, 34, 39]
    
    # Filter to only available epochs
    early_epochs = [e for e in early_epochs if e in epoch_similarities]
    full_epochs = [e for e in full_epochs if e in epoch_similarities]
    
    # Compute mean similarity per concept per epoch (for all epochs)
    mean_sims = {}
    for epoch in epoch_similarities:
        mean_sims[epoch] = epoch_similarities[epoch].mean(axis=0)  # [num_concepts]
    
    # Compute rankings for each epoch (higher similarity = better rank = lower number)
    rankings = {}
    for epoch in epoch_similarities:
        sorted_indices = np.argsort(-mean_sims[epoch])
        rank = np.zeros(len(ALL_CONCEPTS), dtype=int)
        for r, idx in enumerate(sorted_indices):
            rank[idx] = r + 1  # 1-indexed rank
        rankings[epoch] = rank
    
    num_concepts = len(ALL_CONCEPTS)
    
    # Create figure with two subplots side by side
    fig, (ax_early, ax_full) = plt.subplots(1, 2, figsize=(24, 14))
    
    def plot_ranking_subplot(ax, plot_epochs, title_suffix):
        """Helper function to plot ranking evolution on a single axis."""
        # Create x-axis positions for this subplot
        x_positions = []
        x_labels = []
        for i, e in enumerate(plot_epochs):
            x_positions.append(i)  # Use sequential positions
            if e == 'baseline':
                x_labels.append('Base')
            else:
                x_labels.append(f'Ep{e}')
        
        # Plot each concept's ranking trajectory
        for concept_idx, concept in enumerate(ALL_CONCEPTS):
            group = CONCEPT_TO_GROUP[concept]
            color = get_group_color(group)
            
            rank_values = [rankings[e][concept_idx] for e in plot_epochs]
            
            # Line style based on group type (egocentric vs allocentric)
            if group == 'egocentric':
                linestyle = '--'
                alpha = 0.7
            else:
                linestyle = '-'
                alpha = 0.85
            
            ax.plot(x_positions, rank_values, marker='o', markersize=3, 
                    color=color, linestyle=linestyle, alpha=alpha, linewidth=1.2)
        
        # Add concept labels on the right side (at final epoch of this subplot)
        final_rankings = rankings[plot_epochs[-1]]
        sorted_by_final = np.argsort(final_rankings)
        
        for concept_idx in sorted_by_final:
            concept = ALL_CONCEPTS[concept_idx]
            group = CONCEPT_TO_GROUP[concept]
            color = get_group_color(group)
            final_rank = final_rankings[concept_idx]
            
            label = concept[:25] + '...' if len(concept) > 25 else concept
            ax.text(x_positions[-1] + 0.15, final_rank, label, 
                    fontsize=4, va='center', ha='left', color=color)
        
        # Add concept labels on the left side (at first epoch of this subplot)
        first_rankings = rankings[plot_epochs[0]]
        sorted_by_first = np.argsort(first_rankings)
        
        for concept_idx in sorted_by_first:
            concept = ALL_CONCEPTS[concept_idx]
            group = CONCEPT_TO_GROUP[concept]
            color = get_group_color(group)
            first_rank = first_rankings[concept_idx]
            
            label = concept[:25] + '...' if len(concept) > 25 else concept
            ax.text(x_positions[0] - 0.15, first_rank, label,
                    fontsize=4, va='center', ha='right', color=color)
        
        # Invert y-axis so rank 1 is at top
        ax.invert_yaxis()
        ax.set_ylim(num_concepts + 1, 0)
        
        # Set x-axis
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel('Training Progress', fontsize=11)
        ax.set_ylabel('Concept Rank (1 = highest similarity)', fontsize=11)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=num_concepts/2, color='gray', linestyle=':', alpha=0.5)
        
        # Title
        ax.set_title(title_suffix, fontsize=12, fontweight='bold')
    
    # Plot early training (left subplot)
    plot_ranking_subplot(ax_early, early_epochs, 'Early Training (Baseline → Epoch 4)')
    
    # Plot full training (right subplot)
    plot_ranking_subplot(ax_full, full_epochs, 'Full Training (Baseline → Epoch 39)')
    
    # Add shared legend at the bottom using GROUP_COLORS
    legend_elements = [
        Line2D([0], [0], color=GROUP_COLORS['egocentric'], linestyle='--', 
               linewidth=2, label='Egocentric (Self)'),
        Line2D([0], [0], color=GROUP_COLORS['teammate'], linestyle='-',
               linewidth=2, label='Teammate'),
        Line2D([0], [0], color=GROUP_COLORS['enemy'], linestyle='-',
               linewidth=2, label='Enemy'),
        Line2D([0], [0], color=GROUP_COLORS['global'], linestyle='-',
               linewidth=2, label='Global Game State'),
        Line2D([0], [0], color=GROUP_COLORS['spatial'], linestyle='-',
               linewidth=2, label='Spatial'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=5, fontsize=10)
    
    # Main title
    fig.suptitle(f'Sample {sample_idx} - Concept Ranking Evolution ({num_concepts} concepts)\n'
                 f'(Dashed: Egocentric, Solid: Allocentric/Global)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_aggregate_ranking_plot(
    all_sample_rankings: dict,
    save_path: Path,
    epochs: list,
):
    """
    Create an aggregate ranking plot using Borda-style average rank aggregation.
    
    For each concept, compute its average rank across all samples at each epoch,
    then visualize how these aggregate rankings change over training.
    
    Args:
        all_sample_rankings: Dict mapping sample_idx to dict of {epoch: rankings_array}
        save_path: Path to save the figure
        epochs: List of epochs (can include 'baseline' as first element)
    """
    from matplotlib.lines import Line2D
    
    num_samples = len(all_sample_rankings)
    num_concepts = len(ALL_CONCEPTS)
    
    # Compute average rank for each concept at each epoch
    avg_rankings = {}
    for epoch in epochs:
        # Collect ranks from all samples for this epoch
        epoch_ranks = []
        for sample_idx, sample_rankings in all_sample_rankings.items():
            if epoch in sample_rankings:
                epoch_ranks.append(sample_rankings[epoch])
        
        if epoch_ranks:
            # Stack and compute mean across samples
            stacked_ranks = np.stack(epoch_ranks, axis=0)  # [num_samples, num_concepts]
            avg_rankings[epoch] = stacked_ranks.mean(axis=0)  # [num_concepts]
    
    # Re-rank based on average ranks (lower average rank = better = rank 1)
    final_rankings = {}
    for epoch in epochs:
        if epoch in avg_rankings:
            sorted_indices = np.argsort(avg_rankings[epoch])
            rank = np.zeros(num_concepts, dtype=float)
            for r, idx in enumerate(sorted_indices):
                rank[idx] = r + 1
            final_rankings[epoch] = rank
    
    # Define the two epoch splits
    early_epochs = ['baseline', 0, 1, 2, 3, 4]
    full_epochs = ['baseline', 4, 9, 14, 19, 24, 29, 34, 39]
    
    early_epochs = [e for e in early_epochs if e in final_rankings]
    full_epochs = [e for e in full_epochs if e in final_rankings]
    
    # Create figure with two subplots side by side
    fig, (ax_early, ax_full) = plt.subplots(1, 2, figsize=(24, 16))
    
    def plot_aggregate_subplot(ax, plot_epochs, title_suffix):
        """Helper function to plot aggregate ranking evolution."""
        x_positions = []
        x_labels = []
        for i, e in enumerate(plot_epochs):
            x_positions.append(i)
            if e == 'baseline':
                x_labels.append('Base')
            else:
                x_labels.append(f'Ep{e}')
        
        # Plot each concept's average ranking trajectory
        for concept_idx, concept in enumerate(ALL_CONCEPTS):
            group = CONCEPT_TO_GROUP[concept]
            color = get_group_color(group)
            
            rank_values = [final_rankings[e][concept_idx] for e in plot_epochs]
            
            if group == 'egocentric':
                linestyle = '--'
                alpha = 0.7
            else:
                linestyle = '-'
                alpha = 0.85
            
            ax.plot(x_positions, rank_values, marker='o', markersize=3, 
                    color=color, linestyle=linestyle, alpha=alpha, linewidth=1.2)
        
        # Add concept labels on the right side
        final_ranks = final_rankings[plot_epochs[-1]]
        sorted_by_final = np.argsort(final_ranks)
        
        for concept_idx in sorted_by_final:
            concept = ALL_CONCEPTS[concept_idx]
            group = CONCEPT_TO_GROUP[concept]
            color = get_group_color(group)
            final_rank = final_ranks[concept_idx]
            
            label = concept[:25] + '...' if len(concept) > 25 else concept
            ax.text(x_positions[-1] + 0.15, final_rank, label, 
                    fontsize=4, va='center', ha='left', color=color)
        
        # Add concept labels on the left side
        first_ranks = final_rankings[plot_epochs[0]]
        sorted_by_first = np.argsort(first_ranks)
        
        for concept_idx in sorted_by_first:
            concept = ALL_CONCEPTS[concept_idx]
            group = CONCEPT_TO_GROUP[concept]
            color = get_group_color(group)
            first_rank = first_ranks[concept_idx]
            
            label = concept[:25] + '...' if len(concept) > 25 else concept
            ax.text(x_positions[0] - 0.15, first_rank, label,
                    fontsize=4, va='center', ha='right', color=color)
        
        ax.invert_yaxis()
        ax.set_ylim(num_concepts + 1, 0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel('Training Progress', fontsize=11)
        ax.set_ylabel('Average Rank (1 = highest similarity)', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=num_concepts/2, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(title_suffix, fontsize=12, fontweight='bold')
    
    plot_aggregate_subplot(ax_early, early_epochs, 'Early Training (Baseline → Epoch 4)')
    plot_aggregate_subplot(ax_full, full_epochs, 'Full Training (Baseline → Epoch 39)')
    
    # Add shared legend
    legend_elements = [
        Line2D([0], [0], color=GROUP_COLORS['egocentric'], linestyle='--', 
               linewidth=2, label='Egocentric (Self)'),
        Line2D([0], [0], color=GROUP_COLORS['teammate'], linestyle='-',
               linewidth=2, label='Teammate'),
        Line2D([0], [0], color=GROUP_COLORS['enemy'], linestyle='-',
               linewidth=2, label='Enemy'),
        Line2D([0], [0], color=GROUP_COLORS['global'], linestyle='-',
               linewidth=2, label='Global Game State'),
        Line2D([0], [0], color=GROUP_COLORS['spatial'], linestyle='-',
               linewidth=2, label='Spatial'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=5, fontsize=10)
    
    fig.suptitle(f'Aggregate Ranking Evolution (Borda-style, {num_samples} samples, {num_concepts} concepts)\n'
                 f'(Dashed: Egocentric, Solid: Allocentric/Global)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create a summary table of top movers
    return create_ranking_summary(final_rankings, epochs, num_samples)


def create_group_ranking_plot(
    all_sample_rankings: dict,
    save_path: Path,
    epochs: list,
):
    """
    Create a clean aggregate ranking plot showing only 5 curves (one per group).
    Each curve shows the average rank of concepts within that group.
    Also shows shaded regions for min/max range within each group.
    
    Args:
        all_sample_rankings: Dict mapping sample_idx to dict of {epoch: rankings_array}
        save_path: Path to save the figure
        epochs: List of epochs (can include 'baseline' as first element)
    """
    from matplotlib.lines import Line2D
    
    num_samples = len(all_sample_rankings)
    num_concepts = len(ALL_CONCEPTS)
    
    # Compute average rank for each concept at each epoch (across samples)
    avg_rankings = {}
    for epoch in epochs:
        epoch_ranks = []
        for sample_idx, sample_rankings in all_sample_rankings.items():
            if epoch in sample_rankings:
                epoch_ranks.append(sample_rankings[epoch])
        
        if epoch_ranks:
            stacked_ranks = np.stack(epoch_ranks, axis=0)
            avg_rankings[epoch] = stacked_ranks.mean(axis=0)
    
    # For each group, compute mean, min, max rank at each epoch
    group_stats = {group: {'mean': [], 'min': [], 'max': [], 'std': []} 
                   for group in CATEGORY_GROUPS.keys()}
    
    available_epochs = [e for e in epochs if e in avg_rankings]
    
    for epoch in available_epochs:
        for group in CATEGORY_GROUPS.keys():
            # Get indices of concepts in this group
            group_indices = [i for i, c in enumerate(ALL_CONCEPTS) if CONCEPT_TO_GROUP[c] == group]
            group_ranks = avg_rankings[epoch][group_indices]
            
            group_stats[group]['mean'].append(np.mean(group_ranks))
            group_stats[group]['min'].append(np.min(group_ranks))
            group_stats[group]['max'].append(np.max(group_ranks))
            group_stats[group]['std'].append(np.std(group_ranks))
    
    # Define epoch splits
    early_epochs = ['baseline', 0, 1, 2, 3, 4]
    full_epochs = ['baseline', 4, 9, 14, 19, 24, 29, 34, 39]
    
    early_epochs = [e for e in early_epochs if e in avg_rankings]
    full_epochs = [e for e in full_epochs if e in avg_rankings]
    
    # Create figure
    fig, (ax_early, ax_full) = plt.subplots(1, 2, figsize=(16, 8))
    
    def get_epoch_indices(plot_epochs, all_epochs):
        """Get indices into the stats arrays for the given epochs."""
        return [all_epochs.index(e) for e in plot_epochs if e in all_epochs]
    
    def plot_group_subplot(ax, plot_epochs, title_suffix):
        """Plot group-level ranking evolution."""
        x_positions = list(range(len(plot_epochs)))
        x_labels = ['Base' if e == 'baseline' else f'Ep{e}' for e in plot_epochs]
        epoch_indices = get_epoch_indices(plot_epochs, available_epochs)
        
        for group in CATEGORY_GROUPS.keys():
            color = get_group_color(group)
            
            means = [group_stats[group]['mean'][i] for i in epoch_indices]
            mins = [group_stats[group]['min'][i] for i in epoch_indices]
            maxs = [group_stats[group]['max'][i] for i in epoch_indices]
            
            # Line style
            if group == 'egocentric':
                linestyle = '--'
            else:
                linestyle = '-'
            
            # Plot shaded region (min to max)
            ax.fill_between(x_positions, mins, maxs, color=color, alpha=0.15)
            
            # Plot mean line
            ax.plot(x_positions, means, marker='o', markersize=8, 
                    color=color, linestyle=linestyle, linewidth=2.5, 
                    label=f'{group.title()} (n={len([c for c in ALL_CONCEPTS if CONCEPT_TO_GROUP[c] == group])})')
        
        ax.invert_yaxis()
        ax.set_ylim(num_concepts + 10, 0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlabel('Training Progress', fontsize=12)
        ax.set_ylabel('Average Rank (1 = highest similarity)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=num_concepts/2, color='gray', linestyle=':', alpha=0.5, label='_nolegend_')
        ax.set_title(title_suffix, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
    
    plot_group_subplot(ax_early, early_epochs, 'Early Training (Baseline → Epoch 4)')
    plot_group_subplot(ax_full, full_epochs, 'Full Training (Baseline → Epoch 39)')
    
    fig.suptitle(f'Group-Level Ranking Evolution ({num_samples} samples)\n'
                 f'Lines: Mean rank, Shaded: Min-Max range within group', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return group-level summary
    return compute_group_summary(group_stats, available_epochs)


def compute_group_summary(group_stats: dict, epochs: list) -> dict:
    """Compute summary statistics for group ranking changes."""
    summary = {}
    
    if len(epochs) < 2:
        return summary
    
    for group in group_stats.keys():
        baseline_mean = group_stats[group]['mean'][0]
        final_mean = group_stats[group]['mean'][-1]
        change = baseline_mean - final_mean  # positive = rose in ranking
        
        summary[group] = {
            'baseline_mean': baseline_mean,
            'final_mean': final_mean,
            'change': change,
            'baseline_range': (group_stats[group]['min'][0], group_stats[group]['max'][0]),
            'final_range': (group_stats[group]['min'][-1], group_stats[group]['max'][-1]),
        }
    
    return summary


def create_ranking_summary(final_rankings: dict, epochs: list, num_samples: int) -> dict:
    """
    Create a summary of ranking changes, identifying top risers and fallers.
    
    Returns:
        Summary dict with top risers and fallers
    """
    if 'baseline' not in final_rankings or epochs[-1] not in final_rankings:
        return {}
    
    baseline_ranks = final_rankings['baseline']
    final_ranks = final_rankings[epochs[-1]]
    
    # Compute rank change (positive = improved/rose, negative = fell)
    rank_changes = baseline_ranks - final_ranks
    
    # Get top risers (biggest positive change = rose most in ranking)
    risers_idx = np.argsort(-rank_changes)[:20]
    # Get top fallers (biggest negative change = fell most in ranking)
    fallers_idx = np.argsort(rank_changes)[:20]
    
    summary = {
        'top_risers': [(ALL_CONCEPTS[i], CONCEPT_TO_GROUP[ALL_CONCEPTS[i]], 
                        int(baseline_ranks[i]), int(final_ranks[i]), int(rank_changes[i])) 
                       for i in risers_idx],
        'top_fallers': [(ALL_CONCEPTS[i], CONCEPT_TO_GROUP[ALL_CONCEPTS[i]], 
                         int(baseline_ranks[i]), int(final_ranks[i]), int(rank_changes[i])) 
                        for i in fallers_idx],
    }
    
    return summary


def main():
    """Main function to generate language visualizations."""
    # Configuration
    experiment_name = "main_ui_cover-siglip2-ui-all-260122-064933-md8t"
    # All epochs for trajectory visualization: baseline, 0, 1, 2, 3, 4, 9, 14, 19, 24, 29, 34, 39
    epochs_to_load = [0, 1, 2, 3, 4, 9, 14, 19, 24, 29, 34, 39]
    final_epoch = 39  # For before/after comparison (baseline vs epoch 39)
    num_samples = 100
    
    # Setup paths
    output_base = Path(get_output_base_path())
    data_base = Path(get_data_base_path())
    artifacts_dir = Path("artifacts") / "language_visualization"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Language Visualization for SigLIP2 Contrastive Training")
    print("=" * 60)
    
    # Load experiment config
    print("\n[1/5] Loading experiment configuration...")
    cfg = load_experiment_config(experiment_name)
    
    # Load data
    print("\n[2/5] Loading test data...")
    df = pl.read_csv(data_base / "labels" / "contrastive.csv", null_values=[])
    df = df.filter(pl.col('partition') == 'test')
    
    # Sample videos
    random.seed(42)
    all_indices = list(range(len(df)))
    sample_indices = random.sample(all_indices, min(num_samples, len(all_indices)))
    sample_indices.sort()
    
    print(f"  Selected {len(sample_indices)} samples from {len(df)} test samples")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Load SigLIP2 model and processor
    print("\n[3/5] Loading SigLIP2 model...")
    siglip_model, processor = load_siglip2_model()
    siglip_model = siglip_model.to(device)
    siglip_model.eval()
    
    # Pre-compute text embeddings (these don't change)
    print(f"  Computing text embeddings for {len(ALL_CONCEPTS)} concepts...")
    text_embeds = get_text_embeddings(siglip_model, processor, ALL_CONCEPTS, device)
    
    # Save baseline (off-the-shelf pretrained) vision weights
    # This is the original pretrained model before any finetuning
    baseline_vision_state = {k: v.clone() for k, v in siglip_model.vision_model.state_dict().items()}
    
    # Load finetuned weights for different epochs
    print("\n[4/5] Loading checkpoint weights...")
    epoch_vision_states = {}
    
    for epoch in epochs_to_load:
        print(f"  Loading epoch {epoch}...")
        state = load_checkpoint_vision_state(experiment_name, epoch)
        if state is not None:
            epoch_vision_states[epoch] = state
    
    # Process each sample and collect rankings for aggregation
    print("\n[5/6] Generating individual sample visualizations...")
    all_sample_rankings = {}  # For Borda-style aggregation
    all_epochs_for_plot = None
    
    for sample_idx in tqdm(sample_indices, desc="Processing samples"):
        row = df.row(sample_idx, named=True)
        match_id = row['match_id']
        round_num = row['round_num']
        start_seconds = row['start_seconds']
        end_seconds = row['end_seconds']
        
        # Use first teammate
        agent_id = row['teammate_0_id']
        if agent_id is None:
            continue
        
        # Load video
        video_path = construct_video_path(cfg, match_id, str(agent_id), round_num)
        try:
            video_result = load_video_clip(cfg, video_path, start_seconds, end_seconds)
            video_clip = video_result['video']
        except Exception as e:
            print(f"  Warning: Failed to load video for sample {sample_idx}: {e}")
            continue
        
        # Create sample directory
        sample_dir = artifacts_dir / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Compute similarities with baseline (off-the-shelf pretrained) model
        replace_vision_encoder_weights(siglip_model, baseline_vision_state)
        baseline_sims = compute_similarities_for_video(
            video_clip, siglip_model, processor, text_embeds, device
        )
        
        # 2. Create before/after plot: baseline vs epoch 39
        if final_epoch in epoch_vision_states and epoch_vision_states[final_epoch] is not None:
            replace_vision_encoder_weights(siglip_model, epoch_vision_states[final_epoch])
            finetuned_sims = compute_similarities_for_video(
                video_clip, siglip_model, processor, text_embeds, device
            )
            
            # Create before/after plot with difference matrix
            create_before_after_plot(
                baseline_sims, finetuned_sims, video_clip, sample_idx,
                sample_dir / "before_after.png"
            )
        
        # 3. Compute similarities for each epoch (for evolution visualization)
        # Start with baseline as the first point
        epoch_similarities = {'baseline': baseline_sims}
        
        available_epochs = [e for e in epochs_to_load if e in epoch_vision_states and epoch_vision_states[e] is not None]
        
        for epoch in available_epochs:
            replace_vision_encoder_weights(siglip_model, epoch_vision_states[epoch])
            epoch_similarities[epoch] = compute_similarities_for_video(
                video_clip, siglip_model, processor, text_embeds, device
            )
        
        # Create trajectory plots with baseline + all epochs
        all_epochs_for_plot = ['baseline'] + available_epochs
        
        if len(available_epochs) >= 1:
            # Create epoch evolution plot
            create_epoch_evolution_plot(
                epoch_similarities,
                video_clip, sample_idx,
                sample_dir / "epoch_evolution.png",
                all_epochs_for_plot
            )
            
            # Create concept trajectory plot
            create_concept_trajectory_plot(
                epoch_similarities,
                sample_idx,
                sample_dir / "concept_trajectories.png",
                all_epochs_for_plot
            )
            
            # Create ranking change plot (all concepts)
            create_ranking_change_plot(
                epoch_similarities,
                sample_idx,
                sample_dir / "ranking_evolution.png",
                all_epochs_for_plot
            )
            
            # Collect rankings for aggregation
            # Compute rankings for this sample at each epoch
            sample_rankings = {}
            for epoch in all_epochs_for_plot:
                mean_sim = epoch_similarities[epoch].mean(axis=0)  # [num_concepts]
                sorted_indices = np.argsort(-mean_sim)
                rank = np.zeros(len(ALL_CONCEPTS), dtype=int)
                for r, idx in enumerate(sorted_indices):
                    rank[idx] = r + 1
                sample_rankings[epoch] = rank
            all_sample_rankings[sample_idx] = sample_rankings
        
        # Restore baseline weights for next sample
        replace_vision_encoder_weights(siglip_model, baseline_vision_state)
    
    # Create aggregate ranking plot
    print("\n[6/6] Generating aggregate ranking visualization...")
    if all_sample_rankings and all_epochs_for_plot:
        aggregate_dir = artifacts_dir / "sample_aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        
        # Create detailed per-concept aggregate plot (all 272 concepts)
        summary = create_aggregate_ranking_plot(
            all_sample_rankings,
            aggregate_dir / "ranking_evolution_all_concepts.png",
            all_epochs_for_plot
        )
        
        # Create clean group-level plot (5 curves only)
        group_summary = create_group_ranking_plot(
            all_sample_rankings,
            aggregate_dir / "ranking_evolution_by_group.png",
            all_epochs_for_plot
        )
        
        # Save summary to text file
        with open(aggregate_dir / "ranking_summary.txt", 'w') as f:
            f.write(f"Aggregate Ranking Summary ({len(all_sample_rankings)} samples, {len(ALL_CONCEPTS)} concepts)\n")
            f.write("=" * 70 + "\n\n")
            
            # Group-level summary
            f.write("GROUP-LEVEL RANKING CHANGES:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Group':<15} {'Baseline':>12} {'Final':>12} {'Change':>12} {'Direction':<15}\n")
            f.write("-" * 70 + "\n")
            if group_summary:
                for group in ['egocentric', 'teammate', 'enemy', 'global', 'spatial']:
                    if group in group_summary:
                        gs = group_summary[group]
                        direction = "ROSE ^" if gs['change'] > 0 else "FELL v" if gs['change'] < 0 else "SAME"
                        f.write(f"{group:<15} {gs['baseline_mean']:>12.1f} {gs['final_mean']:>12.1f} {gs['change']:>+12.1f} {direction:<15}\n")
            
            if summary:
                f.write("\n\nTOP 20 RISERS (concepts that rose most in ranking after training):\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'Concept':<40} {'Group':<12} {'Base':>6} {'Final':>6} {'Change':>8}\n")
                f.write("-" * 70 + "\n")
                for concept, group, base_rank, final_rank, change in summary['top_risers']:
                    f.write(f"{concept[:38]:<40} {group:<12} {base_rank:>6} {final_rank:>6} {change:>+8}\n")
                
                f.write("\n\nTOP 20 FALLERS (concepts that fell most in ranking after training):\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'Concept':<40} {'Group':<12} {'Base':>6} {'Final':>6} {'Change':>8}\n")
                f.write("-" * 70 + "\n")
                for concept, group, base_rank, final_rank, change in summary['top_fallers']:
                    f.write(f"{concept[:38]:<40} {group:<12} {base_rank:>6} {final_rank:>6} {change:>+8}\n")
        
        print(f"  Summary saved to: {aggregate_dir / 'ranking_summary.txt'}")
    
    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {artifacts_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
