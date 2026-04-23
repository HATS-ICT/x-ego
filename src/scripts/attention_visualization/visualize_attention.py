import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from omegaconf import OmegaConf
import platform
import cv2  # type: ignore

from src.models.contrastive_model import ContrastiveModel
from src.dataset.dataset_utils import init_video_processor, construct_video_path, load_video_clip, transform_video
from src.utils.env_utils import get_output_base_path, get_data_base_path

if platform.system() == 'Windows':
    import pathlib._local
    pathlib._local.PosixPath = pathlib._local.WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath


def load_vision_model_eager(model_type):
    from transformers import AutoModel, AutoConfig, Dinov2Model, SiglipVisionModel, CLIPVisionModel
    from src.models.modules.video_encoder import MODEL_TYPE_TO_PRETRAINED
    pretrained = MODEL_TYPE_TO_PRETRAINED[model_type]
    if model_type == 'siglip2':
        config = AutoConfig.from_pretrained(pretrained)
        config.vision_config._attn_implementation = 'eager'
        model = SiglipVisionModel.from_pretrained(pretrained, config=config.vision_config, attn_implementation='eager')
        return model
    elif model_type == 'clip':
        model = CLIPVisionModel.from_pretrained(pretrained, attn_implementation='eager')
        return model
    elif model_type in ['dinov2', 'dinov3']:
        return Dinov2Model.from_pretrained(pretrained, attn_implementation='eager')
    elif model_type == 'vjepa2':
        model = AutoModel.from_pretrained(pretrained, attn_implementation='eager')
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_checkpoint_model(experiment_name, epoch, cfg):
    output_base = Path(get_output_base_path())
    exp_dir = output_base / experiment_name
    checkpoint_dir = exp_dir / "checkpoint"
    ckpt_files = list(checkpoint_dir.glob(f"*-e{epoch:02d}-*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found for epoch {epoch} in {checkpoint_dir}")
    checkpoint_path = ckpt_files[0]
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    state_dict = checkpoint['state_dict']
    state_dict = ContrastiveModel._strip_orig_mod_prefix(state_dict)
    model = ContrastiveModel(cfg)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model


def get_attention_weights(vision_model, pixel_values, model_type):
    batch_size, num_frames, channels, height, width = pixel_values.shape
    
    if model_type == 'vjepa2':
        # VJEPA2 processes video natively with spatiotemporal patches
        outputs = vision_model(pixel_values_videos=pixel_values, output_attentions=True, skip_predictor=True)
        attentions = outputs.attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
        print(f"VJEPA2 Attention shape: {attentions.shape}, input: {pixel_values.shape}")
        return attentions
    else:
        # Frame-by-frame models (CLIP, SigLIP2, DINOv2)
        frames = pixel_values.view(-1, channels, height, width)
        outputs = vision_model(pixel_values=frames, output_attentions=True)
        attentions = outputs.attentions[-1]
        print(f"Attention shape: {attentions.shape}, input: {frames.shape}")
        return attentions


def visualize_vjepa2_attention_on_frame(original_frame, attention, frame_idx, num_time_patches, num_patches_h, num_patches_w, num_input_frames=None):
    """
    Visualize VJEPA2 attention on a specific frame.
    
    VJEPA2 uses spatiotemporal patches. The attention is over all patches (T * H * W).
    We extract the spatial attention for a specific temporal position.
    
    Args:
        original_frame: Original video frame (C, H, W) in [0, 255] or [0, 1] range
        attention: Attention weights (num_heads, total_patches, total_patches)
        frame_idx: Which frame we're visualizing (0 to num_frames-1)
        num_time_patches: Number of temporal patches
        num_patches_h: Number of patches in height
        num_patches_w: Number of patches in width
        num_input_frames: Total number of input frames (for proper mapping)
    """
    # attention shape: (num_heads, total_patches, total_patches)
    # total_patches = num_time_patches * num_patches_h * num_patches_w
    attn_mean = attention.mean(dim=0)  # (total_patches, total_patches)
    
    num_spatial_patches = num_patches_h * num_patches_w
    
    # Map frame_idx to the closest temporal patch
    # VJEPA2 uses tubelet_size=2, so each time patch covers 2 frames
    if num_input_frames is not None:
        # Proper mapping: frame_idx / num_input_frames * num_time_patches
        time_patch_idx = min(int(frame_idx / num_input_frames * num_time_patches), num_time_patches - 1)
    else:
        time_patch_idx = min(frame_idx // 2, num_time_patches - 1)
    
    # Get the spatial patch indices for this temporal position
    # VJEPA2 flattens in T, H, W order
    start_idx = time_patch_idx * num_spatial_patches
    end_idx = start_idx + num_spatial_patches
    
    # Get attention received by spatial patches at this time step
    # Sum attention from all patches to these spatial patches
    spatial_attn = attn_mean[:, start_idx:end_idx].sum(dim=0)  # (num_spatial_patches,)
    
    spatial_attn = spatial_attn.reshape(num_patches_h, num_patches_w).cpu().numpy()
    spatial_attn = (spatial_attn - spatial_attn.min()) / (spatial_attn.max() - spatial_attn.min() + 1e-8)
    
    # Convert original frame to numpy
    frame_np = original_frame.permute(1, 2, 0).cpu().numpy()
    if frame_np.max() <= 1.0:
        frame_np = (frame_np * 255).astype(np.uint8)
    else:
        frame_np = frame_np.astype(np.uint8)
    
    h, w = frame_np.shape[:2]
    attn_resized = cv2.resize(spatial_attn, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(frame_np, 0.6, heatmap, 0.4, 0)
    
    return overlay


def visualize_attention_on_frame(original_frame, attention, num_patches_h, num_patches_w, model_type='dinov2'):
    """
    Visualize attention on a frame.
    
    Args:
        original_frame: Original video frame (C, H, W) in [0, 255] or [0, 1] range (NOT normalized)
        attention: Attention weights from the model (num_heads, num_tokens, num_tokens)
        num_patches_h: Number of patches in height
        num_patches_w: Number of patches in width
        model_type: Type of model ('dinov2'/'clip' has CLS token, 'siglip2' does not)
    """
    # attention shape: (num_heads, num_tokens, num_tokens)
    attn_mean = attention.mean(dim=0)  # (num_tokens, num_tokens)
    num_tokens = attn_mean.shape[0]
    num_patches = num_patches_h * num_patches_w
    
    if model_type in ['dinov2', 'clip']:
        # DINOv2 and CLIP have CLS token at position 0
        # Extract attention from CLS token to all patch tokens
        if num_tokens == num_patches + 1:
            cls_attn = attn_mean[0, 1:]  # CLS attending to patches
        else:
            cls_attn = attn_mean.mean(dim=0)
    else:
        # SigLIP2 has no CLS token - all tokens are patch tokens
        # Use mean attention received by each patch (how much each patch is attended to)
        # attn_mean[i, j] = attention from token i to token j
        # Sum over dim=0 gives total attention received by each token
        cls_attn = attn_mean.sum(dim=0)  # Total attention each patch receives
    
    cls_attn = cls_attn[:num_patches].reshape(num_patches_h, num_patches_w).cpu().numpy()
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    
    # Convert original frame to numpy (should be in [0, 255] or [0, 1] range, NOT normalized)
    frame_np = original_frame.permute(1, 2, 0).cpu().numpy()
    if frame_np.max() <= 1.0:
        frame_np = (frame_np * 255).astype(np.uint8)
    else:
        frame_np = frame_np.astype(np.uint8)
    
    h, w = frame_np.shape[:2]
    attn_resized = cv2.resize(cls_attn, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(frame_np, 0.6, heatmap, 0.4, 0)
    
    return overlay


def process_experiment(experiment_name, short_name, epoch, sample_indices, df, pretrained_vision_models):
    output_base = Path(get_output_base_path())
    data_base = Path(get_data_base_path())
    exp_dir = output_base / experiment_name
    hparam_path = exp_dir / "hparam.yaml"
    
    cfg = OmegaConf.load(hparam_path)
    cfg.data.partition = "test"
    # Disable random mask for visualization
    if "random_mask" in cfg.data:
        cfg.data.random_mask.enable = False
    else:
        cfg.data.random_mask = {"enable": False}
    
    path_cfg = OmegaConf.create({
        'path': {
            'exp': str(exp_dir),
            'ckpt': str(exp_dir / "checkpoint"),
            'plots': str(exp_dir / "plots"),
            'data': str(data_base),
            'src': str(Path(__file__).parent.parent.parent),
            'output': str(output_base)
        },
        'data': {
            'label_path': str(data_base / cfg.data.labels_folder / cfg.data.labels_filename),
            'video_base_path': str(data_base / cfg.data.video_folder)
        }
    })
    cfg = OmegaConf.merge(cfg, path_cfg)
    
    model = load_checkpoint_model(experiment_name, epoch, cfg)
    video_processor, processor_type = init_video_processor(cfg)
    
    model_type = cfg.model.encoder.model_type
    
    # Load finetuned vision model (with checkpoint weights)
    finetuned_vision_model = load_vision_model_eager(model_type)
    
    if model_type == 'vjepa2':
        # VJEPA2 uses encoder.* structure
        encoder_state = model.video_encoder.video_encoder.vision_model.encoder.state_dict()
        new_state = {f"encoder.{k}": v for k, v in encoder_state.items()}
        finetuned_vision_model.load_state_dict(new_state, strict=False)
    elif model_type == 'siglip2':
        encoder_state = model.video_encoder.video_encoder.vision_model.state_dict()
        new_state = {f"vision_model.{k}": v for k, v in encoder_state.items()}
        finetuned_vision_model.load_state_dict(new_state)
    elif model_type == 'clip':
        # CLIPVisionModel expects keys with "vision_model." prefix
        encoder_state = model.video_encoder.video_encoder.vision_model.state_dict()
        new_state = {f"vision_model.{k}": v for k, v in encoder_state.items()}
        finetuned_vision_model.load_state_dict(new_state)
    else:
        encoder_state = model.video_encoder.video_encoder.vision_model.state_dict()
        finetuned_vision_model.load_state_dict(encoder_state)
    finetuned_vision_model.eval()
    finetuned_vision_model.cuda()
    
    # Get pretrained (off-the-shelf) vision model
    pretrained_vision_model = pretrained_vision_models[model_type]
    
    if model_type == 'dinov2':
        num_patches_h, num_patches_w = 16, 16
        num_time_patches = None
    elif model_type == 'siglip2':
        num_patches_h, num_patches_w = 14, 14
        num_time_patches = None
    elif model_type == 'clip':
        # CLIP ViT-B/32 uses 32x32 patches on 224x224 images -> 7x7 patches
        num_patches_h, num_patches_w = 7, 7
        num_time_patches = None
    elif model_type == 'vjepa2':
        # VJEPA2 uses 16x16 spatial patches and tubelet_size=2 for temporal
        # For 256x256 input: 16x16 spatial patches
        num_patches_h, num_patches_w = 16, 16
        # Time patches depend on input frames, will be computed dynamically
        num_time_patches = None  # Will be set per video
    else:
        num_patches_h, num_patches_w = 14, 14
        num_time_patches = None
    
    artifacts_dir = Path("artifacts") / "attention_visualization" / short_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    for sample_idx in sample_indices:
        row = df.row(sample_idx, named=True)
        match_id = row['match_id']
        round_num = row['round_num']
        start_seconds = row['start_seconds']
        end_seconds = row['end_seconds']
        num_alive_teammates = row['num_alive_teammates']
        
        # Iterate over all alive teammates
        for teammate_idx in range(num_alive_teammates):
            agent_id = row[f'teammate_{teammate_idx}_id']
            if agent_id is None:
                continue
            
            video_path = construct_video_path(cfg, match_id, str(agent_id), round_num)
            video_clip_result = load_video_clip(cfg, video_path, start_seconds, end_seconds)
            # load_video_clip returns a dict when random_mask config exists
            if isinstance(video_clip_result, dict):
                video_clip = video_clip_result['video']
            else:
                video_clip = video_clip_result
            video_features = transform_video(video_processor, processor_type, video_clip)
            video_tensor = video_features.unsqueeze(0).cuda()
            
            with torch.no_grad():
                finetuned_attentions = get_attention_weights(finetuned_vision_model, video_tensor, model_type)
                pretrained_attentions = get_attention_weights(pretrained_vision_model, video_tensor, model_type)
            
            num_frames = video_tensor.shape[1]
            frame_indices = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]
            
            # Determine row labels based on model type
            model_display_names = {
                'siglip2': 'SigLIP',
                'dinov2': 'DINOv2',
                'clip': 'CLIP',
                'vjepa2': 'V-JEPA2',
            }
            display_name = model_display_names.get(model_type, model_type.upper())
            row_labels = ["Original", f"{display_name}\nContrastive", f"{display_name}\nOriginal"]
            
            fig, axes = plt.subplots(3, 5, figsize=(8, 4))
            plt.subplots_adjust(wspace=-0.35, hspace=0.01, left=0.10, right=0.99, top=0.94, bottom=0.01)
            
            for i, frame_idx in enumerate(frame_indices):
                frame = video_clip[frame_idx]
                frame_np = frame.permute(1, 2, 0).cpu().numpy()
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
                else:
                    frame_np = frame_np.astype(np.uint8)
                
                # Row 0: Original frames (only show Frame labels on first row)
                axes[0, i].imshow(frame_np)
                axes[0, i].set_title(f"Frame {frame_idx}", fontsize=8, pad=1)
                axes[0, i].axis('off')
                
                if model_type == 'vjepa2':
                    # VJEPA2 returns attention over all spatiotemporal patches
                    # finetuned_attentions shape: [batch_size, num_heads, total_patches, total_patches]
                    # We need to compute num_time_patches from the attention shape
                    total_patches = finetuned_attentions.shape[-1]
                    num_spatial = num_patches_h * num_patches_w
                    computed_time_patches = total_patches // num_spatial
                    
                    # Row 1: Finetuned model attention
                    finetuned_attention = finetuned_attentions[0]  # [num_heads, total_patches, total_patches]
                    finetuned_overlay = visualize_vjepa2_attention_on_frame(
                        video_clip[frame_idx], finetuned_attention, frame_idx, 
                        computed_time_patches, num_patches_h, num_patches_w, num_frames
                    )
                    axes[1, i].imshow(finetuned_overlay)
                    axes[1, i].axis('off')
                    
                    # Row 2: Pretrained (off-the-shelf) model attention
                    pretrained_attention = pretrained_attentions[0]  # [num_heads, total_patches, total_patches]
                    pretrained_overlay = visualize_vjepa2_attention_on_frame(
                        video_clip[frame_idx], pretrained_attention, frame_idx,
                        computed_time_patches, num_patches_h, num_patches_w, num_frames
                    )
                    axes[2, i].imshow(pretrained_overlay)
                    axes[2, i].axis('off')
                else:
                    # Frame-by-frame models (CLIP, SigLIP2, DINOv2)
                    # Row 1: Finetuned model attention
                    finetuned_attention = finetuned_attentions[frame_idx]
                    finetuned_overlay = visualize_attention_on_frame(video_clip[frame_idx], finetuned_attention, num_patches_h, num_patches_w, model_type)
                    axes[1, i].imshow(finetuned_overlay)
                    axes[1, i].axis('off')
                    
                    # Row 2: Pretrained (off-the-shelf) model attention
                    pretrained_attention = pretrained_attentions[frame_idx]
                    pretrained_overlay = visualize_attention_on_frame(video_clip[frame_idx], pretrained_attention, num_patches_h, num_patches_w, model_type)
                    axes[2, i].imshow(pretrained_overlay)
                    axes[2, i].axis('off')
            
            # Add vertical row labels on the left
            for row_idx, label in enumerate(row_labels):
                axes[row_idx, 0].text(-0.08, 0.5, label, fontsize=7, rotation=90, va='center', ha='center', transform=axes[row_idx, 0].transAxes)
            
            save_path = artifacts_dir / f"sample{sample_idx}_teammate{teammate_idx}.pdf"
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
            plt.close()
            print(f"  Saved: {save_path}")


def load_pretrained_vision_models(model_types=None):
    """Load off-the-shelf pretrained vision models (without finetuning).
    
    Args:
        model_types: List of model types to load. If None, loads all supported models.
    """
    if model_types is None:
        model_types = ['dinov2', 'siglip2', 'clip', 'vjepa2']
    
    pretrained_models = {}
    
    for model_type in model_types:
        print(f"Loading pretrained {model_type}...")
        pretrained_models[model_type] = load_vision_model_eager(model_type)
        pretrained_models[model_type].eval()
        pretrained_models[model_type].cuda()
    
    return pretrained_models


def process_experiment_selected(experiment_name, short_name, epoch, selected_samples, df, pretrained_vision_models, output_format='svg'):
    """
    Process only selected samples for a specific experiment.
    
    Args:
        selected_samples: List of (sample_idx, teammate_idx) tuples
        output_format: 'svg' or 'pdf'
    """
    output_base = Path(get_output_base_path())
    data_base = Path(get_data_base_path())
    exp_dir = output_base / experiment_name
    hparam_path = exp_dir / "hparam.yaml"
    
    cfg = OmegaConf.load(hparam_path)
    cfg.data.partition = "test"
    # Disable random mask for visualization
    if "random_mask" in cfg.data:
        cfg.data.random_mask.enable = False
    else:
        cfg.data.random_mask = {"enable": False}
    
    path_cfg = OmegaConf.create({
        'path': {
            'exp': str(exp_dir),
            'ckpt': str(exp_dir / "checkpoint"),
            'plots': str(exp_dir / "plots"),
            'data': str(data_base),
            'src': str(Path(__file__).parent.parent.parent),
            'output': str(output_base)
        },
        'data': {
            'label_path': str(data_base / cfg.data.labels_folder / cfg.data.labels_filename),
            'video_base_path': str(data_base / cfg.data.video_folder)
        }
    })
    cfg = OmegaConf.merge(cfg, path_cfg)
    
    model = load_checkpoint_model(experiment_name, epoch, cfg)
    video_processor, processor_type = init_video_processor(cfg)
    
    model_type = cfg.model.encoder.model_type
    
    # Load finetuned vision model (with checkpoint weights)
    finetuned_vision_model = load_vision_model_eager(model_type)
    
    if model_type == 'vjepa2':
        encoder_state = model.video_encoder.video_encoder.vision_model.encoder.state_dict()
        new_state = {f"encoder.{k}": v for k, v in encoder_state.items()}
        finetuned_vision_model.load_state_dict(new_state, strict=False)
    elif model_type == 'siglip2':
        encoder_state = model.video_encoder.video_encoder.vision_model.state_dict()
        new_state = {f"vision_model.{k}": v for k, v in encoder_state.items()}
        finetuned_vision_model.load_state_dict(new_state)
    elif model_type == 'clip':
        # CLIPVisionModel expects keys with "vision_model." prefix
        encoder_state = model.video_encoder.video_encoder.vision_model.state_dict()
        new_state = {f"vision_model.{k}": v for k, v in encoder_state.items()}
        finetuned_vision_model.load_state_dict(new_state)
    else:
        encoder_state = model.video_encoder.video_encoder.vision_model.state_dict()
        finetuned_vision_model.load_state_dict(encoder_state)
    finetuned_vision_model.eval()
    finetuned_vision_model.cuda()
    
    pretrained_vision_model = pretrained_vision_models[model_type]
    
    if model_type == 'dinov2':
        num_patches_h, num_patches_w = 16, 16
    elif model_type == 'siglip2':
        num_patches_h, num_patches_w = 14, 14
    elif model_type == 'clip':
        num_patches_h, num_patches_w = 7, 7
    elif model_type == 'vjepa2':
        num_patches_h, num_patches_w = 16, 16
    else:
        num_patches_h, num_patches_w = 14, 14
    
    artifacts_dir = Path("artifacts") / "attention_visualization" / f"{short_name}_selected"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    for sample_idx, teammate_idx in selected_samples:
        row = df.row(sample_idx, named=True)
        match_id = row['match_id']
        round_num = row['round_num']
        start_seconds = row['start_seconds']
        end_seconds = row['end_seconds']
        
        agent_id = row[f'teammate_{teammate_idx}_id']
        if agent_id is None:
            print(f"  Skipping sample{sample_idx}_teammate{teammate_idx}: agent_id is None")
            continue
        
        video_path = construct_video_path(cfg, match_id, str(agent_id), round_num)
        video_clip_result = load_video_clip(cfg, video_path, start_seconds, end_seconds)
        if isinstance(video_clip_result, dict):
            video_clip = video_clip_result['video']
        else:
            video_clip = video_clip_result
        video_features = transform_video(video_processor, processor_type, video_clip)
        video_tensor = video_features.unsqueeze(0).cuda()
        
        with torch.no_grad():
            finetuned_attentions = get_attention_weights(finetuned_vision_model, video_tensor, model_type)
            pretrained_attentions = get_attention_weights(pretrained_vision_model, video_tensor, model_type)
        
        num_frames = video_tensor.shape[1]
        frame_indices = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]
        
        model_display_names = {
            'siglip2': 'SigLIP',
            'dinov2': 'DINOv2',
            'clip': 'CLIP',
            'vjepa2': 'V-JEPA2',
        }
        display_name = model_display_names.get(model_type, model_type.upper())
        row_labels = ["Original", f"{display_name}\nContrastive", f"{display_name}\nOriginal"]
        
        fig, axes = plt.subplots(3, 5, figsize=(8, 4))
        plt.subplots_adjust(wspace=-0.35, hspace=0.01, left=0.10, right=0.99, top=0.94, bottom=0.01)
        
        for i, frame_idx in enumerate(frame_indices):
            frame = video_clip[frame_idx]
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
            
            axes[0, i].imshow(frame_np)
            axes[0, i].set_title(f"Frame {frame_idx}", fontsize=8, pad=1)
            axes[0, i].axis('off')
            
            if model_type == 'vjepa2':
                total_patches = finetuned_attentions.shape[-1]
                num_spatial = num_patches_h * num_patches_w
                computed_time_patches = total_patches // num_spatial
                
                finetuned_attention = finetuned_attentions[0]
                finetuned_overlay = visualize_vjepa2_attention_on_frame(
                    video_clip[frame_idx], finetuned_attention, frame_idx, 
                    computed_time_patches, num_patches_h, num_patches_w, num_frames
                )
                axes[1, i].imshow(finetuned_overlay)
                axes[1, i].axis('off')
                
                pretrained_attention = pretrained_attentions[0]
                pretrained_overlay = visualize_vjepa2_attention_on_frame(
                    video_clip[frame_idx], pretrained_attention, frame_idx,
                    computed_time_patches, num_patches_h, num_patches_w, num_frames
                )
                axes[2, i].imshow(pretrained_overlay)
                axes[2, i].axis('off')
            else:
                finetuned_attention = finetuned_attentions[frame_idx]
                finetuned_overlay = visualize_attention_on_frame(video_clip[frame_idx], finetuned_attention, num_patches_h, num_patches_w, model_type)
                axes[1, i].imshow(finetuned_overlay)
                axes[1, i].axis('off')
                
                pretrained_attention = pretrained_attentions[frame_idx]
                pretrained_overlay = visualize_attention_on_frame(video_clip[frame_idx], pretrained_attention, num_patches_h, num_patches_w, model_type)
                axes[2, i].imshow(pretrained_overlay)
                axes[2, i].axis('off')
        
        for row_idx, label in enumerate(row_labels):
            axes[row_idx, 0].text(-0.08, 0.5, label, fontsize=7, rotation=90, va='center', ha='center', transform=axes[row_idx, 0].transAxes)
        
        save_path = artifacts_dir / f"sample{sample_idx}_teammate{teammate_idx}.{output_format}"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.02, format=output_format)
        plt.close()
        print(f"  Saved: {save_path}")


if __name__ == "__main__":
    import random
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected', action='store_true', help='Generate only selected samples in SVG format')
    args = parser.parse_args()
    
    experiments = [
        ("main_ui_cover-dinov2-ui-all-260122-035704-my8c", "dinov2-ui_all", "dinov2"),
        ("main_ui_cover-siglip2-ui-all-260122-064933-md8t", "siglip2-ui_all", "siglip2"),
        ("main_ui_cover-vjepa2-ui-all-260122-072237-nrz4", "vjepa2-ui_all", "vjepa2"),
        ("main_ui_cover-clip-ui-all-260124-084053-wxbo", "clip-ui_all", "clip"),
    ]
    
    epoch = 39
    
    data_base = Path(get_data_base_path())
    df = pl.read_csv(data_base / "labels" / "contrastive.csv", null_values=[])
    df = df.filter(pl.col('partition') == 'test')
    
    if args.selected:
        # Selected samples from the paper: (sample_idx, teammate_idx)
        selected_samples = [
            (1323, 2),
            (375, 3),
            (5242, 3),
            (53, 2),
            (569, 1),
            (712, 1),
            (912, 1),
            (916, 0),
        ]
        
        print(f"Generating {len(selected_samples)} selected samples in SVG format")
        
        model_types_needed = list(set(exp[2] for exp in experiments))
        pretrained_vision_models = load_pretrained_vision_models(model_types_needed)
        
        for experiment_name, short_name, _ in experiments:
            print(f"Processing: {short_name}")
            process_experiment_selected(experiment_name, short_name, epoch, selected_samples, df, pretrained_vision_models, output_format='svg')
    else:
        num_samples = 200
        
        # Randomly sample indices
        random.seed(42)
        all_indices = list(range(len(df)))
        sample_indices = random.sample(all_indices, min(num_samples, len(all_indices)))
        sample_indices.sort()
        
        print(f"Sampling {len(sample_indices)} random samples from {len(df)} test samples")
        
        model_types_needed = list(set(exp[2] for exp in experiments))
        pretrained_vision_models = load_pretrained_vision_models(model_types_needed)
        
        for experiment_name, short_name, _ in experiments:
            print(f"Processing: {short_name}")
            process_experiment(experiment_name, short_name, epoch, sample_indices, df, pretrained_vision_models)
