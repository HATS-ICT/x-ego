"""
Training utilities for X-EGO project.
Handles PyTorch Lightning callbacks, loggers, and debugging tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from lightning.pytorch.loggers import WandbLogger


def setup_callbacks(cfg):
    """Setup Lightning callbacks"""
    callbacks = []
    
    # Checkpoint callbacks (required if present in cfg)
    if 'checkpoint' in cfg:
        checkpoint_cfg = cfg.checkpoint
        if 'dirpath' in checkpoint_cfg:
            # Only create checkpoint callbacks if dirpath is set (not None)
            if 'epoch' in checkpoint_cfg and 'step' in checkpoint_cfg:
                epoch_cfg = checkpoint_cfg.epoch
                epoch_callback = ModelCheckpoint(
                    dirpath=checkpoint_cfg.dirpath,
                    filename=epoch_cfg.filename,
                    monitor=epoch_cfg.monitor,
                    mode=epoch_cfg.mode,
                    save_top_k=epoch_cfg.save_top_k,
                    save_last=epoch_cfg.save_last,
                    auto_insert_metric_name=epoch_cfg.auto_insert_metric_name,
                    save_on_train_epoch_end=epoch_cfg.save_on_train_epoch_end,
                    every_n_epochs=epoch_cfg.every_n_epochs
                )
                callbacks.append(epoch_callback)
                
                # Step-based checkpoint callback
                step_cfg = checkpoint_cfg.step
                step_callback = ModelCheckpoint(
                    dirpath=checkpoint_cfg.dirpath,
                    filename=step_cfg.filename,
                    monitor=step_cfg.monitor,
                    mode=step_cfg.mode,
                    save_top_k=step_cfg.save_top_k,
                    save_last=step_cfg.save_last,
                    auto_insert_metric_name=step_cfg.auto_insert_metric_name,
                    save_on_train_epoch_end=step_cfg.save_on_train_epoch_end,
                    every_n_train_steps=step_cfg.every_n_train_steps
                )
                callbacks.append(step_callback)
                
            else:
                checkpoint_callback = ModelCheckpoint(
                    dirpath=checkpoint_cfg.dirpath,
                    filename=checkpoint_cfg.filename,
                    monitor=checkpoint_cfg.monitor,
                    mode=checkpoint_cfg.mode,
                    save_top_k=checkpoint_cfg.save_top_k,
                    save_last=checkpoint_cfg.save_last,
                    auto_insert_metric_name=checkpoint_cfg.auto_insert_metric_name,
                    save_on_train_epoch_end=checkpoint_cfg.save_on_train_epoch_end,
                    every_n_train_steps=checkpoint_cfg.every_n_train_steps,
                    every_n_epochs=checkpoint_cfg.every_n_epochs
                )
                callbacks.append(checkpoint_callback)
    
    # Early stopping (optional)
    if 'early_stopping' in cfg:
        early_stop_cfg = cfg.early_stopping
        early_stopping = EarlyStopping(
            monitor=early_stop_cfg.monitor,
            patience=early_stop_cfg.patience,
            mode=early_stop_cfg.mode
        )
        callbacks.append(early_stopping)
    
    callbacks.append(ModelSummary(max_depth=2))
    
    return callbacks


def setup_logger(cfg):
    """Setup WandB logger with automatic tags"""
    if 'wandb' not in cfg:
        return None
        
    wandb_cfg = cfg.wandb
    
    if not wandb_cfg.enabled:
        return None
    
    # Only create logger if save_dir is set (not None)
    if wandb_cfg.save_dir is None:
        return None
    
    # Automatically generate tags
    tags = list(wandb_cfg.tags) if wandb_cfg.tags else []
    
    # Add task type tag based on exp_name
    exp_name = cfg.meta.exp_name if hasattr(cfg.meta, 'exp_name') else cfg.meta.run_name
    if 'contrastive' in exp_name.lower():
        tags.append('task:contrastive')
        tags.append('stage:1')
    elif 'enemy-nowcast' in exp_name or 'enemy-now' in exp_name:
        tags.append('task:enemy-now')
    elif 'teammate' in exp_name:
        tags.append('task:teammate-now')
    
    # Add contrastive tag (only for contrastive task, downstream uses stage1_checkpoint)
    if 'contrastive' in cfg.model:
        contrastive_enabled = cfg.model.contrastive.enable
        tags.append('yes_contra' if contrastive_enabled else 'no_contra')
    elif cfg.model.get('stage1_checkpoint'):
        tags.append('yes_contra')  # Using pretrained contrastive encoder
    else:
        tags.append('no_contra')  # Baseline with off-the-shelf encoder
    
    # Initialize wandb
    logger = WandbLogger(
        project=wandb_cfg.project,
        name=wandb_cfg.name,  # name can be None for auto-generation
        group=wandb_cfg.group,
        tags=tags,
        notes=wandb_cfg.notes,
        save_dir=wandb_cfg.save_dir
    )
    
    return logger


# ===========================
# Debug and Visualization Tools
# ===========================

def _to_numpy(x):
    """Convert torch tensor or numpy array to numpy on CPU."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def debug_batch_plot(batch, model, max_examples=4):
    """
    Visualize batch examples with inputs (multi-agent videos) and labels.
    
    For each example, shows:
    1) Multi-agent video frames (one frame per agent)
    2) Labels visualization: Heatmap showing which places each player occupies
    
    Args:
        batch: Batch dictionary containing 'video' and labels
        model: Model instance to access cfg and task-specific info
        max_examples: Number of examples to visualize (default: 4)
    """
    
    # Get batch info
    batch_size = batch["video"].shape[0]
    num_examples = min(max_examples, batch_size)
    
    # Get videos: [B, A, T, C, H, W]
    videos = batch["video"][:num_examples]
    num_agents = videos.shape[1]
    
    # Get labels based on task type
    if 'enemy_locations' in batch:
        labels = batch['enemy_locations'][:num_examples]
        label_type = 'enemy'
    elif 'teammate_locations' in batch:
        labels = batch['teammate_locations'][:num_examples]
        label_type = 'teammate'
    else:
        print("Warning: No labels found in batch")
        return
    
    cfg = model.cfg
    
    # Create separate figure for each example
    for i in range(num_examples):
        # Create figure with side-by-side layout: video (left, large) and labels (right, compact)
        fig = plt.figure(figsize=(22, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)
        
        # Left: video frames (larger)
        video_ax = fig.add_subplot(gs[0, 0])
        _plot_multi_agent_video(videos[i], video_ax, num_agents)
        team_info = batch['pov_team_side'][i] if 'pov_team_side' in batch else 'N/A'
        video_ax.set_title(f"Example {i+1}: Multi-Agent Video (Team: {team_info})", 
                          fontsize=14, fontweight='bold')
        
        # Right: labels (more compact)
        label_ax = fig.add_subplot(gs[0, 1])
        _plot_labels(labels[i], label_ax, cfg, label_type)
        
        plt.tight_layout()
        plt.show()
        plt.close()


def _plot_multi_agent_video(video_tensor, ax, num_agents):
    """
    Plot multi-agent video frames by stacking 5 frames per agent.
    
    Args:
        video_tensor: [A, T, C, H, W] tensor
        ax: Matplotlib axis
        num_agents: Number of agents
    """
    # video_tensor: [A, T, C, H, W]
    A, T, C, H, W = video_tensor.shape
    
    # Select 5 evenly spaced frames from each agent's video
    num_frames_to_show = min(5, T)
    frame_indices = np.linspace(0, T - 1, num_frames_to_show, dtype=int)
    
    # Get frames for all agents: [A, num_frames_to_show, C, H, W]
    video = video_tensor[:, frame_indices, :, :, :]
    video = _to_numpy(video)
    
    # Stack frames for each agent
    agent_strips = []
    for a in range(A):
        agent_video = video[a]  # [num_frames_to_show, C, H, W]
        
        # Normalize and process each frame
        frame_list = []
        for t in range(num_frames_to_show):
            frame = agent_video[t]  # [C, H, W]
            
            # Normalize per-frame robustly
            fr = frame.astype(np.float32)
            if fr.max() <= 1.0 and fr.min() >= -1.0 and fr.min() < 0.0:
                # Values in [-1, 1] range, rescale to [0, 1]
                fr = (fr + 1.0) / 2.0
            else:
                # Min-max normalization per frame
                fmin, fmax = fr.min(), fr.max()
                if fmax > fmin:
                    fr = (fr - fmin) / (fmax - fmin)
                else:
                    fr = np.zeros_like(fr)
            
            # Convert to [H, W, C]
            fr = np.transpose(fr, (1, 2, 0))
            
            # Handle grayscale
            if fr.shape[-1] == 1:
                fr = np.repeat(fr, 3, axis=-1)
            
            frame_list.append(fr)
        
        # Concatenate frames horizontally: [H, num_frames_to_show*W, 3]
        agent_strip = np.concatenate(frame_list, axis=1)
        agent_strips.append(agent_strip)
    
    # Stack all agents vertically: [A*H, num_frames_to_show*W, 3]
    combined_frame = np.concatenate(agent_strips, axis=0)
    combined_frame = np.clip(combined_frame, 0.0, 1.0)
    
    ax.imshow(combined_frame)
    ax.axis('off')
    
    # Add agent labels
    for a in range(A):
        ax.text(num_frames_to_show * W // 2, H * a + H - 10, f'Agent {a+1}', 
               ha='center', va='top', color='yellow', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))


def _plot_labels(labels_tensor, ax, cfg, label_type='enemy'):
    """
    Plot multi-label classification labels.
    
    Args:
        labels_tensor: Label tensor [num_places] - binary vector indicating occupied places
        ax: Matplotlib axis
        cfg: Configuration object
        label_type: 'enemy' or 'teammate'
    """
    labels = _to_numpy(labels_tensor)
    
    # Labels shape: [num_places] - binary vector indicating occupied places
    place_names = cfg.place_names if hasattr(cfg, 'place_names') else None
    
    if place_names is None:
        # Fallback: use indices
        place_names = [f'Place {i}' for i in range(len(labels))]
    
    # Reshape to [1, num_places] for visualization as a single row heatmap
    labels_2d = labels.reshape(1, -1)
    
    # Create heatmap
    im = ax.imshow(labels_2d, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Place', fontsize=12)
    ax.set_ylabel('Occupied', fontsize=12)
    ax.set_title(f'{label_type.capitalize()} Location Labels (Multi-Label Classification)\nBinary vector: 1 = at least one player present', 
                fontsize=12, fontweight='bold')
    
    # Set y-axis
    ax.set_yticks([0])
    ax.set_yticklabels(['Any Player'])
    
    # Only show place names if there aren't too many
    ax.set_xticks(range(len(place_names)))
    ax.set_xticklabels(list(place_names), rotation=45, ha='right', fontsize=8)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Presence (0=Absent, 1=Present)')
    
    # Add text annotations for positive labels
    for place_idx in range(len(labels)):
        if labels[place_idx] > 0.5:
            ax.text(place_idx, 0, '✓', ha='center', va='center', 
                   color='black', fontsize=10, fontweight='bold')
    
def setup_test_model_with_dataset_info(cfg, datamodule, test_model):
    """Setup test model with dataset-specific information"""
    # Model setup logic can be added here if needed in the future
    pass
    
    
def print_task_info(cfg, datamodule, task_name):
    """Print task-specific information"""
    # Contrastive learning task
    if task_name == 'contrastive':
        print(f"Video encoder: {cfg.model.encoder.video.model_type}")
        print(f"Projection dimension: {cfg.model.encoder.proj_dim}")
        print(f"Allow variable agents: {cfg.data.allow_variable_agents}")
        print(f"Minimum agents: {cfg.data.min_agents}")
        print(f"Contrastive temperature init: {cfg.model.contrastive.logit_scale_init}")
        print(f"Contrastive bias init: {cfg.model.contrastive.logit_bias_init}")
    # Linear probing task
    elif task_name == 'downstream':
        print(f"Task ID: {cfg.task.task_id}")
        print(f"ML form: {cfg.task.ml_form}")
        print(f"Num classes: {cfg.task.num_classes}")
        print(f"Output dim: {cfg.task.output_dim}")
        print(f"Encoder frozen: {cfg.model.encoder.video.freeze_backbone}")
        if cfg.model.stage1_checkpoint:
            print(f"Stage 1 checkpoint: {cfg.model.stage1_checkpoint}")
    # Common info for location prediction tasks
    elif 'location' in task_name:
        print(f"Number of agents: {cfg.data.num_pov_agents}")
        print(f"Agent fusion method: {cfg.model.agent_fusion.method}")
        
        # Get num_places from datamodule after setup and update cfg
        cfg.num_places = datamodule.num_places
        print(f"Number of places: {cfg.num_places}")