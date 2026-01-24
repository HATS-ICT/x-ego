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
    from transformers import AutoModel, AutoConfig, Dinov2Model, SiglipVisionModel
    from src.models.modules.video_encoder import MODEL_TYPE_TO_PRETRAINED
    pretrained = MODEL_TYPE_TO_PRETRAINED[model_type]
    if model_type == 'siglip2':
        config = AutoConfig.from_pretrained(pretrained)
        config.vision_config._attn_implementation = 'eager'
        model = SiglipVisionModel.from_pretrained(pretrained, config=config.vision_config, attn_implementation='eager')
        return model
    elif model_type == 'clip':
        model = AutoModel.from_pretrained(pretrained, attn_implementation='eager')
        return model.vision_model
    elif model_type in ['dinov2', 'dinov3']:
        return Dinov2Model.from_pretrained(pretrained, attn_implementation='eager')
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
    frames = pixel_values.view(-1, channels, height, width)
    outputs = vision_model(pixel_values=frames, output_attentions=True)
    attentions = outputs.attentions[-1]
    print(f"Attention shape: {attentions.shape}, input: {frames.shape}")
    return attentions


def visualize_attention_on_frame(original_frame, attention, num_patches_h, num_patches_w, model_type='dinov2'):
    """
    Visualize attention on a frame.
    
    Args:
        original_frame: Original video frame (C, H, W) in [0, 255] or [0, 1] range (NOT normalized)
        attention: Attention weights from the model (num_heads, num_tokens, num_tokens)
        num_patches_h: Number of patches in height
        num_patches_w: Number of patches in width
        model_type: Type of model ('dinov2' has CLS token, 'siglip2' does not)
    """
    # attention shape: (num_heads, num_tokens, num_tokens)
    attn_mean = attention.mean(dim=0)  # (num_tokens, num_tokens)
    num_tokens = attn_mean.shape[0]
    num_patches = num_patches_h * num_patches_w
    
    if model_type == 'dinov2':
        # DINOv2 has CLS token at position 0
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
    encoder_state = model.video_encoder.video_encoder.vision_model.state_dict()
    if model_type == 'siglip2':
        new_state = {f"vision_model.{k}": v for k, v in encoder_state.items()}
        finetuned_vision_model.load_state_dict(new_state)
    else:
        finetuned_vision_model.load_state_dict(encoder_state)
    finetuned_vision_model.eval()
    finetuned_vision_model.cuda()
    
    # Get pretrained (off-the-shelf) vision model
    pretrained_vision_model = pretrained_vision_models[model_type]
    
    if model_type == 'dinov2':
        num_patches_h, num_patches_w = 16, 16
    elif model_type == 'siglip2':
        num_patches_h, num_patches_w = 14, 14
    else:
        num_patches_h, num_patches_w = 14, 14
    
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
            video_clip = load_video_clip(cfg, video_path, start_seconds, end_seconds)
            video_features = transform_video(video_processor, processor_type, video_clip)
            video_tensor = video_features.unsqueeze(0).cuda()
            
            with torch.no_grad():
                finetuned_attentions = get_attention_weights(finetuned_vision_model, video_tensor, model_type)
                pretrained_attentions = get_attention_weights(pretrained_vision_model, video_tensor, model_type)
            
            num_frames = video_tensor.shape[1]
            frame_indices = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]
            
            # Determine row labels based on model type
            if model_type == 'siglip2':
                row_labels = ["Original", "SigLIP\nContrastive", "SigLIP\nOriginal"]
            else:
                row_labels = ["Original", "DINOv2\nContrastive", "DINOv2\nOriginal"]
            
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


def load_pretrained_vision_models():
    """Load off-the-shelf pretrained vision models (without finetuning)."""
    pretrained_models = {}
    
    # Load pretrained dinov2
    print("Loading pretrained dinov2...")
    pretrained_models['dinov2'] = load_vision_model_eager('dinov2')
    pretrained_models['dinov2'].eval()
    pretrained_models['dinov2'].cuda()
    
    # Load pretrained siglip2
    print("Loading pretrained siglip2...")
    pretrained_models['siglip2'] = load_vision_model_eager('siglip2')
    pretrained_models['siglip2'].eval()
    pretrained_models['siglip2'].cuda()
    
    return pretrained_models


if __name__ == "__main__":
    import random
    
    experiments = [
        ("main_ui_cover-dinov2-ui-all-260122-035704-my8c", "dinov2-ui_all"),
        ("main_ui_cover-siglip2-ui-all-260122-064933-md8t", "siglip2-ui_all"),
    ]
    
    epoch = 39
    num_samples = 200
    
    data_base = Path(get_data_base_path())
    df = pl.read_csv(data_base / "labels" / "contrastive.csv", null_values=[])
    df = df.filter(pl.col('partition') == 'test')
    
    # Randomly sample indices
    random.seed(42)
    all_indices = list(range(len(df)))
    sample_indices = random.sample(all_indices, min(num_samples, len(all_indices)))
    sample_indices.sort()  # Sort for reproducibility in output naming
    
    print(f"Sampling {len(sample_indices)} random samples from {len(df)} test samples")
    
    # Load pretrained models once (shared across experiments)
    pretrained_vision_models = load_pretrained_vision_models()
    
    for experiment_name, short_name in experiments:
        print(f"Processing: {short_name}")
        process_experiment(experiment_name, short_name, epoch, sample_indices, df, pretrained_vision_models)
