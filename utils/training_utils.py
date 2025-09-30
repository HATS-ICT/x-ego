"""
Training utilities for X-EGO project.
Handles PyTorch Lightning callbacks, loggers, and debugging tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from lightning.pytorch.loggers import WandbLogger


def setup_callbacks(config):
    """Setup Lightning callbacks"""
    callbacks = []
    
    # Checkpoint callbacks (required if present in config)
    if 'checkpoint' in config:
        checkpoint_config = config['checkpoint']
        # Only create checkpoint callbacks if dirpath is set (not None)
        if checkpoint_config.get('dirpath') is not None:
            if 'epoch' in checkpoint_config and 'step' in checkpoint_config:
                epoch_config = checkpoint_config['epoch']
                epoch_callback = ModelCheckpoint(
                    dirpath=checkpoint_config['dirpath'],
                    filename=epoch_config['filename'],
                    monitor=epoch_config['monitor'],
                    mode=epoch_config['mode'],
                    save_top_k=epoch_config['save_top_k'],
                    save_last=epoch_config['save_last'],
                    auto_insert_metric_name=epoch_config['auto_insert_metric_name'],
                    save_on_train_epoch_end=epoch_config['save_on_train_epoch_end'],
                    every_n_epochs=epoch_config['every_n_epochs']
                )
                callbacks.append(epoch_callback)
                
                # Step-based checkpoint callback
                step_config = checkpoint_config['step']
                step_callback = ModelCheckpoint(
                    dirpath=checkpoint_config['dirpath'],
                    filename=step_config['filename'],
                    monitor=step_config['monitor'],
                    mode=step_config['mode'],
                    save_top_k=step_config['save_top_k'],
                    save_last=step_config['save_last'],
                    auto_insert_metric_name=step_config['auto_insert_metric_name'],
                    save_on_train_epoch_end=step_config['save_on_train_epoch_end'],
                    every_n_train_steps=step_config['every_n_train_steps']
                )
                callbacks.append(step_callback)
                
            else:
                checkpoint_callback = ModelCheckpoint(
                    dirpath=checkpoint_config['dirpath'],
                    filename=checkpoint_config['filename'],
                    monitor=checkpoint_config['monitor'],
                    mode=checkpoint_config['mode'],
                    save_top_k=checkpoint_config['save_top_k'],
                    save_last=checkpoint_config['save_last'],
                    auto_insert_metric_name=checkpoint_config['auto_insert_metric_name'],
                    save_on_train_epoch_end=checkpoint_config['save_on_train_epoch_end'],
                    every_n_train_steps=checkpoint_config.get('every_n_train_steps'),
                    every_n_epochs=checkpoint_config.get('every_n_epochs')
                )
                callbacks.append(checkpoint_callback)
    
    # Early stopping (optional)
    if 'early_stopping' in config:
        early_stop_config = config['early_stopping']
        early_stopping = EarlyStopping(
            monitor=early_stop_config['monitor'],
            patience=early_stop_config['patience'],
            mode=early_stop_config['mode']
        )
        callbacks.append(early_stopping)
    
    callbacks.append(ModelSummary(max_depth=2))
    
    return callbacks


def setup_logger(config):
    """Setup WandB logger"""
    if 'wandb' not in config:
        return None
        
    wandb_config = config['wandb']
    
    if not wandb_config['enabled']:
        return None
    
    # Only create logger if save_dir is set (not None)
    if wandb_config.get('save_dir') is None:
        return None
    
    # Initialize wandb
    logger = WandbLogger(
        project=wandb_config['project'],
        name=wandb_config.get('name'),  # name can be None for auto-generation
        tags=wandb_config['tags'],
        notes=wandb_config['notes'],
        save_dir=wandb_config['save_dir']
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


def _stack_20_frames(video_20_tchw):
    """
    video_20_tchw: shape (T=20, C=3, H, W) or (20, H, W, C)
    Returns a single H x (20*W) x 3 image by concatenating frames along width.
    """
    v = _to_numpy(video_20_tchw)
    # Ensure shape (20, 3, H, W)
    if v.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape {v.shape}")
    if v.shape[0] != 20:
        raise ValueError(f"Expected 20 frames, got {v.shape[0]}")

    # If (20, H, W, C), move to (20, C, H, W)
    if v.shape[-1] in (1, 3) and v.shape[1] not in (1, 3):
        v = np.transpose(v, (0, 3, 1, 2))  # (T, C, H, W)

    if v.shape[1] not in (1, 3):
        raise ValueError(f"Expected channels=1 or 3, got {v.shape[1]}")

    # Normalize per-frame to [0,1] robustly
    vv = []
    for t in range(20):
        frame = v[t]  # (C, H, W)
        # If values look like [-1,1], rescale; else min-max per frame
        fr = frame.astype(np.float32)
        if fr.max() <= 1.0 and fr.min() >= -1.0 and fr.min() < 0.0:
            fr = (fr + 1.0) / 2.0
        else:
            fmin, fmax = fr.min(), fr.max()
            if fmax > fmin:
                fr = (fr - fmin) / (fmax - fmin)
            else:
                fr = np.zeros_like(fr)
        # To HWC
        fr = np.transpose(fr, (1, 2, 0))
        # If grayscale, repeat to 3 channels
        if fr.shape[-1] == 1:
            fr = np.repeat(fr, 3, axis=-1)
        vv.append(fr)

    # Concatenate along width
    stacked = np.concatenate(vv, axis=1)  # H x (20*W) x 3
    stacked = np.clip(stacked, 0.0, 1.0)
    return stacked


def debug_batch_plot(batch):
    """
    Shows 4 examples; each example has:
    1) Top: 20 video frames stacked horizontally as one image
    2) Middle: audio (e.g., spectrogram) with imshow
    3) Bottom: raw text in a text-only axis
    """
    videos = batch["video"][:4]   # expected (B, T=20, C, H, W) or (B, 20, H, W, C)
    audios = batch["audio"][:4]   # expected (B, 80, 3000) or similar 2D per example
    texts  = batch["raw_text"][:4]

    B = 4
    # Create a single figure using a 12-row GridSpec (3 rows per example)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(B * 3, 1, height_ratios=[3, 2, 1] * B)

    for i in range(B):
        # --- Video (stack 20 frames) ---
        ax_v = fig.add_subplot(gs[i*3 + 0, 0])
        stacked_img = _stack_20_frames(videos[i])
        ax_v.imshow(stacked_img)
        ax_v.set_title(f"Example {i+1} — Video (20 frames stacked)")
        ax_v.axis("off")

        # --- Audio ---
        ax_a = fig.add_subplot(gs[i*3 + 1, 0])
        A = _to_numpy(audios[i])
        if A.ndim != 2:
            # Try to squeeze extra dims if present
            A = np.squeeze(A)
            if A.ndim != 2:
                raise ValueError(f"Audio item {i} should be 2D, got shape {A.shape}")
        im = ax_a.imshow(A, aspect="auto", origin="lower")
        ax_a.set_ylabel("Freq bins")
        ax_a.set_xlabel("Time")
        ax_a.set_title(f"Example {i+1} — Audio")
        fig.colorbar(im, ax=ax_a, fraction=0.015, pad=0.02)

        # --- Text ---
        ax_t = fig.add_subplot(gs[i*3 + 2, 0])
        ax_t.axis("off")
        txt = texts[i]
        if not isinstance(txt, str):
            txt = str(txt)
        ax_t.text(
            0.01, 0.9, f"Example {i+1} — Text",
            fontsize=12, va="top", ha="left", transform=ax_t.transAxes
        )
        ax_t.text(
            0.01, 0.75, txt,
            fontsize=11, va="top", ha="left", transform=ax_t.transAxes,
            wrap=True
        )

    plt.tight_layout()
    plt.show()