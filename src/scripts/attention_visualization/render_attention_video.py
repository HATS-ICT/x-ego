"""
Render attention visualization as a video with three views:
- Left: Original video
- Middle: CECL (finetuned checkpoint model) attention overlay
- Right: Off-the-shelf pretrained model attention overlay

Usage:
    python -m src.scripts.attention_visualization.render_attention_video \
        --video_path "data/video_306x306_4fps/1-2d0bb14d-3c29-44be-9337-898da398d5b4-1-1/76561198092542032/round_3.mp4" \
        --model_type dinov2 \
        --experiment_name "main_ui_cover-dinov2-ui-all-260122-035704-my8c" \
        --epoch 39
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import platform
if platform.system() == 'Windows':
    import pathlib._local
    pathlib._local.PosixPath = pathlib._local.WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath

import argparse
import torch
import numpy as np
import cv2
from decord import VideoReader, cpu
from tqdm import tqdm
from omegaconf import OmegaConf

from src.models.contrastive_model import ContrastiveModel
from src.utils.env_utils import get_output_base_path, get_data_base_path
from src.dataset.dataset_utils import apply_all_ui_mask
from src.scripts.attention_visualization.visualize_attention import (
    DEFAULT_EXPERIMENT,
    load_vision_model_eager,
    get_attention_weights,
    visualize_attention_on_frame,
    visualize_vjepa2_attention_on_frame,
)


# Default experiment names for each model type
DEFAULT_EXPERIMENTS = {
    'dinov2': "main_ui_cover-dinov2-ui-all-260122-035704-my8c",
    'siglip2': DEFAULT_EXPERIMENT,
    'vjepa2': "main_ui_cover-vjepa2-ui-all-260122-072237-nrz4",
    'clip': "main_ui_cover-clip-ui-all-260124-084053-wxbo",
}


def get_patch_dimensions(model_type: str) -> tuple[int, int]:
    """Get the number of patches in height and width for each model type."""
    patch_dims = {
        'dinov2': (16, 16),
        'siglip2': (14, 14),
        'clip': (7, 7),
        'vjepa2': (16, 16),
    }
    return patch_dims.get(model_type, (14, 14))


def load_video_frames(video_path: str, apply_ui_mask: bool = True) -> tuple[np.ndarray, float]:
    """
    Load all frames from a video file.
    
    Args:
        video_path: Path to video file
        apply_ui_mask: Whether to apply UI mask (matching checkpoint training)
    
    Returns:
        Tuple of (frames array [T, H, W, C], fps)
    """
    decoder = VideoReader(str(video_path), ctx=cpu(0))
    fps = decoder.get_avg_fps()
    num_frames = len(decoder)
    frames = decoder.get_batch(list(range(num_frames))).asnumpy()
    
    if apply_ui_mask:
        # Convert to tensor [T, C, H, W] for apply_all_ui_mask
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        frames_tensor = apply_all_ui_mask(frames_tensor)
        # Convert back to numpy [T, H, W, C]
        frames = frames_tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    
    return frames, fps


def preprocess_frames_for_model(frames: np.ndarray, model_type: str) -> torch.Tensor:
    """Preprocess frames for the vision model."""
    from transformers import AutoImageProcessor, VJEPA2VideoProcessor
    from src.models.modules.video_encoder import MODEL_TYPE_TO_PRETRAINED
    
    pretrained_model = MODEL_TYPE_TO_PRETRAINED[model_type]
    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    
    if model_type == 'vjepa2':
        processor = VJEPA2VideoProcessor.from_pretrained(pretrained_model)
        processed = processor(videos=frames_tensor, return_tensors="pt")
        return processed.pixel_values_videos.squeeze(0)
    else:
        processor = AutoImageProcessor.from_pretrained(pretrained_model, use_fast=True)
        processed = processor(images=frames_tensor, return_tensors="pt")
        return processed.pixel_values


def load_finetuned_vision_model(experiment_name: str, epoch: int, model_type: str):
    """
    Load a finetuned vision model from a checkpoint.
    
    Returns the vision model with finetuned weights loaded.
    """
    output_base = Path(get_output_base_path())
    data_base = Path(get_data_base_path())
    exp_dir = output_base / experiment_name
    hparam_path = exp_dir / "hparam.yaml"
    
    # Load config
    cfg = OmegaConf.load(hparam_path)
    cfg.data.partition = "test"
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
    
    # Load checkpoint
    checkpoint_dir = exp_dir / "checkpoint"
    ckpt_files = list(checkpoint_dir.glob(f"*-e{epoch:02d}-*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found for epoch {epoch} in {checkpoint_dir}")
    checkpoint_path = ckpt_files[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint['state_dict']
    state_dict = ContrastiveModel._strip_orig_mod_prefix(state_dict)
    model = ContrastiveModel(cfg)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load vision model with eager attention
    finetuned_vision_model = load_vision_model_eager(model_type)
    
    encoder_state = model.video_encoder.video_encoder.vision_model.state_dict()
    missing, unexpected = finetuned_vision_model.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected vision state keys for {model_type}: {unexpected[:5]}")
    if missing:
        print(f"Warning: {len(missing)} missing vision keys while loading eager model")
    
    finetuned_vision_model.eval()
    finetuned_vision_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    return finetuned_vision_model


def create_attention_overlay(
    frame: np.ndarray, 
    attention: torch.Tensor, 
    model_type: str,
    num_patches_h: int,
    num_patches_w: int,
    frame_idx: int = 0,
    num_frames: int = 1,
    num_time_patches: int = None,
) -> np.ndarray:
    """Create attention overlay for a single frame."""
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
    
    if model_type == 'vjepa2' and num_time_patches is not None:
        overlay = visualize_vjepa2_attention_on_frame(
            frame_tensor, attention, frame_idx,
            num_time_patches, num_patches_h, num_patches_w, num_frames
        )
    else:
        overlay = visualize_attention_on_frame(
            frame_tensor, attention, num_patches_h, num_patches_w, model_type
        )
    
    return overlay


def add_title_bar_below(frame: np.ndarray, title: str, bar_height: int = 30, font_scale: float = 0.7) -> np.ndarray:
    """Add a title bar below the frame (doesn't block the video)."""
    h, w = frame.shape[:2]
    
    # Create a black bar for the title
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(title, font, font_scale, thickness)
    
    # Position: centered in the bar
    x = (w - text_width) // 2
    y = (bar_height + text_height) // 2
    
    # Draw text on the bar
    cv2.putText(title_bar, title, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    # Stack frame and title bar vertically
    combined = np.vstack([frame, title_bar])
    
    return combined


def render_attention_video(
    video_path: str,
    model_type: str,
    experiment_name: str,
    epoch: int,
    output_path: str,
    batch_size: int = 16,
    show_titles: bool = True,
) -> None:
    """
    Render a video with three-view attention visualization.
    
    Left: Original video
    Middle: CECL (finetuned) attention overlay
    Right: Off-the-shelf pretrained attention overlay
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading video: {video_path}")
    frames, fps = load_video_frames(video_path)
    num_frames = len(frames)
    print(f"Loaded {num_frames} frames at {fps:.2f} FPS")
    
    num_patches_h, num_patches_w = get_patch_dimensions(model_type)
    
    # Load both models
    print(f"Loading pretrained {model_type} model (off-the-shelf)...")
    pretrained_model = load_vision_model_eager(model_type)
    pretrained_model.eval()
    pretrained_model.to(device)
    
    print(f"Loading finetuned {model_type} model from {experiment_name}...")
    finetuned_model = load_finetuned_vision_model(experiment_name, epoch, model_type)
    
    # Prepare output video writer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    h, w = frames.shape[1:3]
    title_bar_height = 30 if show_titles else 0
    out_width = w * 3  # Three views side by side
    out_height = h + title_bar_height  # Add space for title bar below
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
    
    # Model display names
    model_display_names = {
        'siglip2': 'SigLIP',
        'dinov2': 'DINOv2',
        'clip': 'CLIP',
        'vjepa2': 'V-JEPA2',
    }
    display_name = model_display_names.get(model_type, model_type.upper())
    
    print("Processing frames...")
    
    if model_type == 'vjepa2':
        # VJEPA2 processes all frames together
        print("Preprocessing frames for VJEPA2...")
        video_tensor = preprocess_frames_for_model(frames, model_type)
        video_tensor = video_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            finetuned_attentions = get_attention_weights(finetuned_model, video_tensor, model_type)
            pretrained_attentions = get_attention_weights(pretrained_model, video_tensor, model_type)
        
        # Compute temporal patch info
        total_patches = finetuned_attentions.shape[-1]
        num_spatial = num_patches_h * num_patches_w
        num_time_patches = total_patches // num_spatial
        
        finetuned_attention = finetuned_attentions[0]
        pretrained_attention = pretrained_attentions[0]
        
        for frame_idx in tqdm(range(num_frames), desc="Rendering"):
            frame = frames[frame_idx]
            
            # Create overlays
            finetuned_overlay = create_attention_overlay(
                frame, finetuned_attention, model_type,
                num_patches_h, num_patches_w,
                frame_idx=frame_idx, num_frames=num_frames,
                num_time_patches=num_time_patches,
            )
            pretrained_overlay = create_attention_overlay(
                frame, pretrained_attention, model_type,
                num_patches_h, num_patches_w,
                frame_idx=frame_idx, num_frames=num_frames,
                num_time_patches=num_time_patches,
            )
            
            # Add titles if enabled
            if show_titles:
                frame_with_title = add_title_bar_below(frame, "Original", title_bar_height)
                finetuned_with_title = add_title_bar_below(finetuned_overlay, f"{display_name} + CECL", title_bar_height)
                pretrained_with_title = add_title_bar_below(pretrained_overlay, display_name, title_bar_height)
            else:
                frame_with_title = frame
                finetuned_with_title = finetuned_overlay
                pretrained_with_title = pretrained_overlay
            
            # Combine three views
            combined = np.hstack([frame_with_title, finetuned_with_title, pretrained_with_title])
            
            # Convert RGB to BGR for OpenCV
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            writer.write(combined_bgr)
    else:
        # Frame-by-frame models: process in batches
        for batch_start in tqdm(range(0, num_frames, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, num_frames)
            batch_frames = frames[batch_start:batch_end]
            
            video_tensor = preprocess_frames_for_model(batch_frames, model_type)
            video_tensor = video_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                finetuned_attentions = get_attention_weights(finetuned_model, video_tensor, model_type)
                pretrained_attentions = get_attention_weights(pretrained_model, video_tensor, model_type)
            
            for i, frame_idx in enumerate(range(batch_start, batch_end)):
                frame = frames[frame_idx]
                
                finetuned_overlay = create_attention_overlay(
                    frame, finetuned_attentions[i], model_type,
                    num_patches_h, num_patches_w,
                )
                pretrained_overlay = create_attention_overlay(
                    frame, pretrained_attentions[i], model_type,
                    num_patches_h, num_patches_w,
                )
                
                if show_titles:
                    frame_with_title = add_title_bar_below(frame, "Original", title_bar_height)
                    finetuned_with_title = add_title_bar_below(finetuned_overlay, f"{display_name} + CECL", title_bar_height)
                    pretrained_with_title = add_title_bar_below(pretrained_overlay, display_name, title_bar_height)
                else:
                    frame_with_title = frame
                    finetuned_with_title = finetuned_overlay
                    pretrained_with_title = pretrained_overlay
                
                combined = np.hstack([frame_with_title, finetuned_with_title, pretrained_with_title])
                combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                writer.write(combined_bgr)
    
    writer.release()
    print(f"Saved attention video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render three-view attention visualization video"
    )
    parser.add_argument(
        '--video_path',
        type=str,
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='dinov2',
        choices=['dinov2', 'siglip2', 'clip', 'vjepa2'],
        help='Type of vision model to use'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name for finetuned checkpoint (uses default if not specified)'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=39,
        help='Epoch of checkpoint to load'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to output video'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for frame processing (non-VJEPA2 models)'
    )
    parser.add_argument(
        '--no_titles',
        action='store_true',
        help='Disable title text on frames'
    )
    
    args = parser.parse_args()
    
    # Use default experiment if not specified
    if args.experiment_name is None:
        args.experiment_name = DEFAULT_EXPERIMENTS[args.model_type]
        print(f"Using default experiment: {args.experiment_name}")
    
    # Generate default output path if not specified
    if args.output_path is None:
        video_path = Path(args.video_path)
        video_name = video_path.stem
        args.output_path = f"artifacts/attention_videos/{args.model_type}_{video_name}_3view.mp4"
    
    render_attention_video(
        video_path=args.video_path,
        model_type=args.model_type,
        experiment_name=args.experiment_name,
        epoch=args.epoch,
        output_path=args.output_path,
        batch_size=args.batch_size,
        show_titles=not args.no_titles,
    )


if __name__ == "__main__":
    main()
