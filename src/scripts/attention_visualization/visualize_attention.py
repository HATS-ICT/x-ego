import argparse
import math
import platform
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import cv2  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from src.dataset.dataset_utils import (
    construct_video_path,
    init_video_processor,
    load_video_clip,
    transform_video,
)
from src.models.contrastive_model import ContrastiveModel
from src.models.modules.video_encoder import MODEL_TYPE_TO_PRETRAINED, normalize_model_type
from src.utils.env_utils import get_data_base_path, get_output_base_path

if platform.system() == "Windows":
    import pathlib
    import pathlib._local

    pathlib._local.PosixPath = pathlib._local.WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath


DEFAULT_EXPERIMENT = "main_contra_with_accu-siglip2-mirage-ui-all-260427-080317-eixo"
DEFAULT_ARTIFACT_DIR = Path("artifacts") / "attention_visualization_v3"
COMPACT_TILE_PIXELS = 96
HEATMAP_COLORMAP = cv2.COLORMAP_MAGMA
HEATMAP_ALPHA = 0.35
DEFAULT_SELECTED_SAMPLES = [
    (1323, 2),
    (375, 3),
    (5242, 3),
    (53, 2),
    (569, 1),
    (712, 1),
    (912, 1),
    (916, 0),
]

MODEL_DISPLAY_NAMES = {
    "siglip2": "SigLIP2",
    "dinov2": "DINOv2",
    "dinov3": "DINOv3",
    "clip": "CLIP",
    "vjepa2": "V-JEPA2",
}


def get_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def experiment_dir_from_name(experiment_name: str, experiment_dir: str | None = None) -> Path:
    exp_dir = Path(experiment_dir).expanduser() if experiment_dir else Path(get_output_base_path()) / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    return exp_dir


def resolve_contrastive_label_path(data_base: Path, cfg) -> Path:
    candidates = []
    if "map" in cfg.data:
        candidates.append(data_base / cfg.data.map / cfg.data.labels_folder / cfg.data.labels_filename)
    candidates.append(data_base / cfg.data.labels_folder / cfg.data.labels_filename)

    for path in candidates:
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find contrastive labels. Searched: {searched}")


def load_experiment_config(exp_dir: Path):
    hparam_path = exp_dir / "hparam.yaml"
    if not hparam_path.exists():
        raise FileNotFoundError(f"Experiment hparam.yaml not found: {hparam_path}")

    cfg = OmegaConf.load(hparam_path)
    data_base = Path(get_data_base_path())
    output_base = Path(get_output_base_path())

    cfg.data.partition = "test"
    if "random_mask" in cfg.data:
        cfg.data.random_mask.enable = False
    else:
        cfg.data.random_mask = {"enable": False}

    path_cfg = OmegaConf.create(
        {
            "path": {
                "exp": str(exp_dir),
                "ckpt": str(exp_dir / "checkpoint"),
                "plots": str(exp_dir / "plots"),
                "data": str(data_base),
                "src": str(Path(__file__).parent.parent.parent),
                "output": str(output_base),
            },
            "data": {
                "label_path": str(resolve_contrastive_label_path(data_base, cfg)),
                "video_base_path": str(data_base / cfg.data.map / cfg.data.video_folder),
            },
        }
    )
    cfg = OmegaConf.merge(cfg, path_cfg)
    cfg.model.encoder.model_type = normalize_model_type(cfg.model.encoder.model_type)
    return cfg


def discover_checkpoint_epochs(checkpoint_dir: Path) -> list[int]:
    epochs = set()
    for path in checkpoint_dir.glob("*.ckpt"):
        match = re.search(r"-e(\d+)-", path.name)
        if match:
            epochs.add(int(match.group(1)))
    return sorted(epochs)


def find_checkpoint(checkpoint_dir: Path, epoch: int | None = None, checkpoint_name: str | None = None) -> Path:
    if checkpoint_name:
        checkpoint_path = checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    if epoch is None:
        last_path = checkpoint_dir / "last.ckpt"
        if last_path.exists():
            return last_path
        epochs = discover_checkpoint_epochs(checkpoint_dir)
        if not epochs:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        epoch = epochs[-1]

    ckpt_files = sorted(checkpoint_dir.glob(f"*-e{epoch:02d}-*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found for epoch {epoch} in {checkpoint_dir}")
    return ckpt_files[0]


def parse_epoch_specs(epoch_args: list[str] | None, epoch: int, use_last: bool) -> list[tuple[str, int | None]]:
    if not epoch_args:
        return [("last", None)] if use_last else [(f"epoch_{epoch:02d}", epoch)]

    specs = []
    for raw_value in epoch_args:
        value = raw_value.strip().lower()
        if value == "last":
            specs.append(("last", None))
        else:
            epoch_value = int(value)
            specs.append((f"epoch_{epoch_value:02d}", epoch_value))
    return specs


def valid_selected_samples(selected_samples: list[tuple[int, int]], df_len: int) -> list[tuple[int, int]]:
    valid = []
    skipped = []
    for sample_idx, teammate_idx in selected_samples:
        if 0 <= sample_idx < df_len:
            valid.append((sample_idx, teammate_idx))
        else:
            skipped.append(sample_idx)

    if skipped:
        skipped_text = ", ".join(str(idx) for idx in sorted(set(skipped)))
        print(f"Skipping selected sample indices outside filtered dataframe: {skipped_text}")

    return valid


def load_vision_model_eager(model_type: str):
    from transformers import AutoModel

    model_type = normalize_model_type(model_type)
    pretrained = MODEL_TYPE_TO_PRETRAINED[model_type]
    model = AutoModel.from_pretrained(pretrained, attn_implementation="eager")
    return model.vision_model if hasattr(model, "vision_model") else model


def load_checkpoint_model(
    experiment_name: str,
    epoch: int | None,
    cfg,
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
):
    device = device or get_device()
    if checkpoint_path is None:
        checkpoint_path = find_checkpoint(Path(cfg.path.ckpt), epoch)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ContrastiveModel._strip_orig_mod_prefix(checkpoint["state_dict"])
    model = ContrastiveModel(cfg)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def load_finetuned_vision_model(
    experiment_name: str,
    epoch: int | None,
    cfg,
    checkpoint_path: Path | None,
    device: torch.device,
):
    model = load_checkpoint_model(experiment_name, epoch, cfg, checkpoint_path, device)
    model_type = cfg.model.encoder.model_type
    vision_model = load_vision_model_eager(model_type)
    encoder_state = model.video_encoder.video_encoder.vision_model.state_dict()
    missing, unexpected = vision_model.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected vision state keys for {model_type}: {unexpected[:5]}")
    if missing:
        print(f"  Warning: {len(missing)} missing vision keys while loading eager model")
    vision_model.eval()
    vision_model.to(device)
    return model, vision_model


def get_attention_weights(vision_model, pixel_values: torch.Tensor, model_type: str):
    batch_size, num_frames, channels, height, width = pixel_values.shape

    if model_type == "vjepa2":
        outputs = vision_model(
            pixel_values_videos=pixel_values,
            output_attentions=True,
            skip_predictor=True,
        )
        if outputs.attentions is None:
            raise RuntimeError("VJEPA2 did not return attention weights")
        return outputs.attentions[-1]

    frames = pixel_values.reshape(-1, channels, height, width)
    outputs = vision_model(pixel_values=frames, output_attentions=True)
    if outputs.attentions is None:
        return get_last_layer_attention_by_hook(vision_model, frames)
    return outputs.attentions[-1]


def get_last_attention_module(vision_model):
    encoder = getattr(vision_model, "encoder", None)
    if encoder is None:
        raise RuntimeError(f"Could not find encoder on {type(vision_model).__name__}")

    layers = getattr(encoder, "layers", None) or getattr(encoder, "layer", None)
    if layers is None:
        layers = getattr(vision_model, "layer", None)
    if layers is None:
        raise RuntimeError(f"Could not find transformer layers on {type(vision_model).__name__}")

    last_layer = layers[-1]
    for attr_path in ("self_attn", "attention.attention", "attention"):
        module = last_layer
        try:
            for attr in attr_path.split("."):
                module = getattr(module, attr)
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                return module
        except AttributeError:
            continue

    raise RuntimeError(f"Could not find q/k attention module on {type(last_layer).__name__}")


def manual_self_attention(attn_module, hidden_states: torch.Tensor) -> torch.Tensor:
    query = attn_module.q_proj(hidden_states)
    key = attn_module.k_proj(hidden_states)
    batch_size, seq_len, embed_dim = query.shape
    num_heads = int(getattr(attn_module, "num_heads"))
    head_dim = int(getattr(attn_module, "head_dim", embed_dim // num_heads))

    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    scale = getattr(attn_module, "scale", None)
    if scale is None:
        scale = head_dim**-0.5

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    return F.softmax(scores.float(), dim=-1).to(hidden_states.dtype)


def get_last_layer_attention_by_hook(vision_model, frames: torch.Tensor) -> torch.Tensor:
    attn_module = get_last_attention_module(vision_model)
    captured = {}

    def capture_input(module, args, kwargs):
        if "hidden_states" in kwargs:
            captured["hidden_states"] = kwargs["hidden_states"].detach()
        elif args:
            captured["hidden_states"] = args[0].detach()

    handle = attn_module.register_forward_pre_hook(capture_input, with_kwargs=True)
    try:
        vision_model(pixel_values=frames, output_attentions=False)
    finally:
        handle.remove()

    if "hidden_states" not in captured:
        raise RuntimeError("Could not capture last-layer attention inputs")
    return manual_self_attention(attn_module, captured["hidden_states"])


def compute_temporal_block_attention(attn_module, x: torch.Tensor) -> torch.Tensor:
    batch_size, num_frames, num_tokens, _ = x.shape
    q = rearrange(attn_module.q_proj(x), "b t p (h d) -> (b p) h t d", h=attn_module.num_heads)
    k = rearrange(attn_module.k_proj(x), "b t p (h d) -> (b p) h t d", h=attn_module.num_heads)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn_module.head_dim)

    if attn_module.causal:
        mask = torch.triu(torch.ones(num_frames, num_frames, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, -torch.inf)

    weights = F.softmax(scores, dim=-1)
    return weights.view(batch_size, num_tokens, attn_module.num_heads, num_frames, num_frames)


def get_temporal_attention_weights(contrastive_model, pixel_values: torch.Tensor, model_type: str):
    encoder = contrastive_model.video_encoder.video_encoder
    temporal = getattr(encoder, "temporal", None)
    if temporal is None:
        return None

    batch_size, num_frames, channels, height, width = pixel_values.shape
    frames = pixel_values.reshape(-1, channels, height, width)
    outputs = encoder.vision_model(pixel_values=frames)
    tokens = outputs.last_hidden_state.view(batch_size, num_frames, outputs.last_hidden_state.shape[1], -1)

    last_weights = None
    for block in temporal.blocks:
        last_weights = compute_temporal_block_attention(block.attn, tokens)
        tokens = block(tokens)

    return last_weights


def normalize_map(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    values = np.nan_to_num(values)
    return (values - values.min()) / (values.max() - values.min() + 1e-8)


def frame_to_numpy(original_frame: torch.Tensor) -> np.ndarray:
    frame_np = original_frame.permute(1, 2, 0).detach().cpu().numpy()
    if frame_np.max() <= 1.0:
        frame_np = frame_np * 255
    return frame_np.astype(np.uint8)


def overlay_attention_map(original_frame: torch.Tensor, attention_map: np.ndarray) -> np.ndarray:
    frame_np = frame_to_numpy(original_frame)
    height, width = frame_np.shape[:2]
    attn_resized = cv2.resize(attention_map, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), HEATMAP_COLORMAP)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(frame_np, 1.0 - HEATMAP_ALPHA, heatmap, HEATMAP_ALPHA, 0)


def spatial_attention_to_map(
    attention: torch.Tensor,
    num_patches_h: int,
    num_patches_w: int,
    model_type: str = "siglip2",
    head_index: int | None = None,
) -> np.ndarray:
    if head_index is None:
        attn = attention.mean(dim=0)
    else:
        attn = attention[head_index]

    num_tokens = attn.shape[0]
    num_patches = num_patches_h * num_patches_w
    has_cls_token = model_type in ["dinov2", "dinov3", "clip"] and num_tokens == num_patches + 1

    if has_cls_token:
        patch_attn = attn[0, 1:]
    elif num_tokens == num_patches:
        patch_attn = attn.sum(dim=0)
    else:
        patch_attn = attn.mean(dim=0)[-num_patches:]

    patch_attn = patch_attn[:num_patches].reshape(num_patches_h, num_patches_w)
    return normalize_map(patch_attn)


def temporal_attention_to_map(
    temporal_attention: torch.Tensor,
    frame_idx: int,
    num_patches_h: int,
    num_patches_w: int,
    model_type: str = "siglip2",
    head_index: int | None = None,
) -> np.ndarray:
    # temporal_attention: [tokens, heads, query_frames, key_frames]
    attn = temporal_attention
    if head_index is None:
        attn = attn.mean(dim=1)
    else:
        attn = attn[:, head_index]

    num_patches = num_patches_h * num_patches_w
    if attn.shape[0] == num_patches + 1 and model_type in ["dinov2", "dinov3", "clip"]:
        attn = attn[1:]
    else:
        attn = attn[-num_patches:]

    received_by_frame = attn[:, :, frame_idx].mean(dim=1)
    return normalize_map(received_by_frame.reshape(num_patches_h, num_patches_w))


def visualize_attention_on_frame(
    original_frame,
    attention,
    num_patches_h,
    num_patches_w,
    model_type="siglip2",
    head_index: int | None = None,
):
    attention_map = spatial_attention_to_map(
        attention,
        num_patches_h,
        num_patches_w,
        model_type=model_type,
        head_index=head_index,
    )
    return overlay_attention_map(original_frame, attention_map)


def visualize_temporal_attention_on_frame(
    original_frame,
    temporal_attention,
    frame_idx,
    num_patches_h,
    num_patches_w,
    model_type="siglip2",
    head_index: int | None = None,
):
    attention_map = temporal_attention_to_map(
        temporal_attention,
        frame_idx,
        num_patches_h,
        num_patches_w,
        model_type=model_type,
        head_index=head_index,
    )
    return overlay_attention_map(original_frame, attention_map)


def visualize_vjepa2_attention_on_frame(
    original_frame,
    attention,
    frame_idx,
    num_time_patches,
    num_patches_h,
    num_patches_w,
    num_input_frames=None,
    head_index: int | None = None,
):
    if head_index is None:
        attn = attention.mean(dim=0)
    else:
        attn = attention[head_index]

    num_spatial_patches = num_patches_h * num_patches_w
    if num_input_frames is not None:
        time_patch_idx = min(int(frame_idx / num_input_frames * num_time_patches), num_time_patches - 1)
    else:
        time_patch_idx = min(frame_idx // 2, num_time_patches - 1)

    start_idx = time_patch_idx * num_spatial_patches
    end_idx = start_idx + num_spatial_patches
    spatial_attn = attn[:, start_idx:end_idx].sum(dim=0)
    attention_map = normalize_map(spatial_attn.reshape(num_patches_h, num_patches_w))
    return overlay_attention_map(original_frame, attention_map)


def get_patch_dimensions(model_type: str, vision_model=None) -> tuple[int, int]:
    if vision_model is not None:
        config = vision_model.config
        image_size = getattr(config, "image_size", None)
        patch_size = getattr(config, "patch_size", None)
        if image_size is not None and patch_size is not None:
            if isinstance(image_size, (tuple, list)):
                height, width = image_size
            else:
                height = width = image_size
            if isinstance(patch_size, (tuple, list)):
                patch_h, patch_w = patch_size[-2], patch_size[-1]
            else:
                patch_h = patch_w = patch_size
            return int(height) // int(patch_h), int(width) // int(patch_w)

    patch_dims = {
        "dinov2": (16, 16),
        "dinov3": (16, 16),
        "siglip2": (14, 14),
        "clip": (7, 7),
        "vjepa2": (16, 16),
    }
    return patch_dims.get(model_type, (14, 14))


def resolve_head_indices(attention: torch.Tensor | None, heads_arg: str) -> list[int | None]:
    if attention is None:
        return []

    num_heads = int(attention.shape[0] if attention.dim() == 3 else attention.shape[1])
    requested = heads_arg.strip().lower()
    heads: list[int | None] = []
    if requested in {"mean", "avg"}:
        return [None]
    if requested in {"all", "mean+all", "mean,all"}:
        heads.append(None)
        heads.extend(range(num_heads))
        return heads

    for item in requested.split(","):
        item = item.strip()
        if item in {"mean", "avg"}:
            heads.append(None)
        elif item:
            head = int(item)
            if head < 0 or head >= num_heads:
                raise ValueError(f"Head index {head} is out of range 0..{num_heads - 1}")
            heads.append(head)
    return heads


def head_label(head_index: int | None) -> str:
    return "mean" if head_index is None else f"head_{head_index:02d}"


def compact_head_indices(attention: torch.Tensor, heads_arg: str) -> list[int | None]:
    heads = resolve_head_indices(attention, heads_arg)
    ordered: list[int | None] = []
    for head in heads:
        if head is not None and head not in ordered:
            ordered.append(head)
    if None in heads:
        ordered.append(None)
    return ordered


def row_label_for_head(head_index: int | None) -> str:
    return "mean" if head_index is None else f"head {head_index}"


def frame_indices_for_clip(num_frames: int, frame_indices_arg: str | None = None) -> list[int]:
    if frame_indices_arg:
        indices = []
        for item in frame_indices_arg.split(","):
            item = item.strip()
            if item:
                idx = int(item)
                if idx < 0 or idx >= num_frames:
                    raise ValueError(f"Frame index {idx} is out of range 0..{num_frames - 1}")
                indices.append(idx)
        return indices
    return list(range(num_frames))


def render_attention_grid(
    video_clip: torch.Tensor,
    frame_indices: list[int],
    finetuned_attention,
    pretrained_attention,
    model_type: str,
    num_patches_h: int,
    num_patches_w: int,
    attention_source: str,
    head_index: int | None,
    comparison_label: str | None = None,
) -> plt.Figure:
    display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type.upper())
    head_name = "mean heads" if head_index is None else f"head {head_index}"
    comparison_label = comparison_label or f"{display_name}\nOriginal"
    row_labels = [
        "Original",
        f"{display_name}\nContrastive",
        comparison_label,
    ]

    fig, axes = plt.subplots(3, len(frame_indices), figsize=(8, 4))
    plt.subplots_adjust(wspace=-0.35, hspace=0.01, left=0.10, right=0.99, top=0.90, bottom=0.01)
    fig.suptitle(f"{attention_source} attention - {head_name}", fontsize=9)

    num_frames = video_clip.shape[0]
    for col_idx, frame_idx in enumerate(frame_indices):
        frame = video_clip[frame_idx]
        axes[0, col_idx].imshow(frame_to_numpy(frame))
        axes[0, col_idx].set_title(f"Frame {frame_idx}", fontsize=8, pad=1)
        axes[0, col_idx].axis("off")

        if attention_source == "spatial":
            finetuned_overlay = visualize_attention_on_frame(
                frame,
                finetuned_attention[frame_idx],
                num_patches_h,
                num_patches_w,
                model_type,
                head_index=head_index,
            )
            pretrained_overlay = visualize_attention_on_frame(
                frame,
                pretrained_attention[frame_idx],
                num_patches_h,
                num_patches_w,
                model_type,
                head_index=head_index,
            )
        elif attention_source == "temporal":
            finetuned_overlay = visualize_temporal_attention_on_frame(
                frame,
                finetuned_attention[0],
                frame_idx,
                num_patches_h,
                num_patches_w,
                model_type,
                head_index=head_index,
            )
            pretrained_overlay = visualize_temporal_attention_on_frame(
                frame,
                pretrained_attention[0],
                frame_idx,
                num_patches_h,
                num_patches_w,
                model_type,
                head_index=head_index,
            )
        else:
            total_patches = finetuned_attention.shape[-1]
            computed_time_patches = total_patches // (num_patches_h * num_patches_w)
            finetuned_overlay = visualize_vjepa2_attention_on_frame(
                frame,
                finetuned_attention[0],
                frame_idx,
                computed_time_patches,
                num_patches_h,
                num_patches_w,
                num_frames,
                head_index=head_index,
            )
            pretrained_overlay = visualize_vjepa2_attention_on_frame(
                frame,
                pretrained_attention[0],
                frame_idx,
                computed_time_patches,
                num_patches_h,
                num_patches_w,
                num_frames,
                head_index=head_index,
            )

        axes[1, col_idx].imshow(finetuned_overlay)
        axes[1, col_idx].axis("off")
        axes[2, col_idx].imshow(pretrained_overlay)
        axes[2, col_idx].axis("off")

    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].text(
            -0.08,
            0.5,
            label,
            fontsize=7,
            rotation=90,
            va="center",
            ha="center",
            transform=axes[row_idx, 0].transAxes,
        )
    return fig


def compact_overlay_for_source(
    item: dict,
    frame_idx: int,
    model_type: str,
    num_patches_h: int,
    num_patches_w: int,
    attention_source: str,
    head_index: int | None,
    untrained: bool = False,
) -> np.ndarray:
    frame = item["video_clip"][frame_idx]
    if attention_source == "spatial":
        attention = item["pretrained_spatial"] if untrained else item["finetuned_spatial"]
        overlay = visualize_attention_on_frame(
            frame,
            attention[frame_idx],
            num_patches_h,
            num_patches_w,
            model_type,
            head_index=None if untrained else head_index,
        )
        return cv2.resize(overlay, (COMPACT_TILE_PIXELS, COMPACT_TILE_PIXELS), interpolation=cv2.INTER_AREA)

    if attention_source == "temporal":
        attention = item["pretrained_temporal"] if untrained else item["finetuned_temporal"]
        overlay = visualize_temporal_attention_on_frame(
            frame,
            attention[0],
            frame_idx,
            num_patches_h,
            num_patches_w,
            model_type,
            head_index=None if untrained else head_index,
        )
        return cv2.resize(overlay, (COMPACT_TILE_PIXELS, COMPACT_TILE_PIXELS), interpolation=cv2.INTER_AREA)

    total_patches = item["finetuned_spatial"].shape[-1]
    computed_time_patches = total_patches // (num_patches_h * num_patches_w)
    attention = item["pretrained_spatial"] if untrained else item["finetuned_spatial"]
    overlay = visualize_vjepa2_attention_on_frame(
        frame,
        attention[0],
        frame_idx,
        computed_time_patches,
        num_patches_h,
        num_patches_w,
        item["video_clip"].shape[0],
        head_index=None if untrained else head_index,
    )
    return cv2.resize(overlay, (COMPACT_TILE_PIXELS, COMPACT_TILE_PIXELS), interpolation=cv2.INTER_AREA)


def render_compact_team_grid(
    teammate_items: list[dict],
    frame_indices: list[int],
    head_indices: list[int | None],
    model_type: str,
    num_patches_h: int,
    num_patches_w: int,
    attention_source: str,
) -> plt.Figure:
    rows = [("head", head_index) for head_index in head_indices]
    rows.append(("untrained", None))

    num_rows = len(rows)
    num_cols = len(teammate_items) * len(frame_indices)
    cell_w = 0.50
    cell_h = 0.50
    fig_w = max(8.0, num_cols * cell_w + 1.2)
    fig_h = max(2.5, num_rows * cell_h + 0.8)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_w, fig_h), squeeze=False)
    plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.035, right=0.998, top=0.93, bottom=0.01)

    display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type.upper())
    fig.suptitle(f"{display_name} {attention_source} attention by teammate, frame, and head", fontsize=9)

    for row_idx, (row_type, head_index) in enumerate(rows):
        for teammate_col, item in enumerate(teammate_items):
            for frame_col, frame_idx in enumerate(frame_indices):
                col_idx = teammate_col * len(frame_indices) + frame_col
                ax = axes[row_idx, col_idx]
                ax.imshow(
                    compact_overlay_for_source(
                        item,
                        frame_idx,
                        model_type,
                        num_patches_h,
                        num_patches_w,
                        attention_source,
                        head_index,
                        untrained=row_type == "untrained",
                    )
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                if row_idx == 0:
                    if frame_col == 0:
                        title = f"T{item['teammate_idx']}\nF{frame_idx}"
                    else:
                        title = f"F{frame_idx}"
                    ax.set_title(title, fontsize=5, pad=1)

        row_label = "untrained" if row_type == "untrained" else row_label_for_head(head_index)
        axes[row_idx, 0].text(
            -0.12,
            0.5,
            row_label,
            fontsize=7,
            rotation=90,
            va="center",
            ha="center",
            transform=axes[row_idx, 0].transAxes,
        )

    return fig


def load_teammate_attention_item(
    cfg,
    row: dict,
    teammate_idx: int,
    video_processor,
    processor_type: str,
    finetuned_vision_model,
    pretrained_vision_model,
    contrastive_model,
    baseline_temporal_model,
    model_type: str,
    attention_sources: set[str],
    device: torch.device,
) -> dict | None:
    agent_id = row[f"teammate_{teammate_idx}_id"]
    if agent_id is None:
        return None

    video_path = construct_video_path(cfg, row["match_id"], str(agent_id), row["round_num"])
    video_clip_result = load_video_clip(cfg, video_path, row["start_seconds"], row["end_seconds"])
    video_clip = video_clip_result["video"] if isinstance(video_clip_result, dict) else video_clip_result
    video_features = transform_video(video_processor, processor_type, video_clip)
    video_tensor = video_features.unsqueeze(0).to(device)

    item = {
        "teammate_idx": teammate_idx,
        "agent_id": agent_id,
        "video_clip": video_clip,
    }

    with torch.no_grad():
        if "spatial" in attention_sources or "vjepa2" in attention_sources:
            item["finetuned_spatial"] = get_attention_weights(finetuned_vision_model, video_tensor, model_type)
            item["pretrained_spatial"] = get_attention_weights(pretrained_vision_model, video_tensor, model_type)
        if "temporal" in attention_sources:
            item["finetuned_temporal"] = get_temporal_attention_weights(contrastive_model, video_tensor, model_type)
            if baseline_temporal_model is not None:
                item["pretrained_temporal"] = get_temporal_attention_weights(baseline_temporal_model, video_tensor, model_type)

    return item


def process_compact_team_sample(
    sample_idx: int,
    row: dict,
    cfg,
    artifacts_dir: Path,
    video_processor,
    processor_type: str,
    finetuned_vision_model,
    pretrained_vision_model,
    contrastive_model,
    baseline_temporal_model,
    model_type: str,
    num_patches_h: int,
    num_patches_w: int,
    attention_sources: set[str],
    heads: str,
    output_format: str,
    frame_indices_arg: str | None,
    device: torch.device,
) -> None:
    teammate_items = []
    for teammate_idx in range(row["num_alive_teammates"]):
        item = load_teammate_attention_item(
            cfg,
            row,
            teammate_idx,
            video_processor,
            processor_type,
            finetuned_vision_model,
            pretrained_vision_model,
            contrastive_model,
            baseline_temporal_model,
            model_type,
            attention_sources,
            device,
        )
        if item is not None:
            teammate_items.append(item)

    if not teammate_items:
        print(f"  Skipping sample{sample_idx}: no teammate videos")
        return

    num_frames = teammate_items[0]["video_clip"].shape[0]
    frame_indices = frame_indices_for_clip(num_frames, frame_indices_arg)

    for attention_source in sorted(attention_sources):
        if attention_source == "temporal":
            valid_items = [
                item
                for item in teammate_items
                if item.get("finetuned_temporal") is not None and item.get("pretrained_temporal") is not None
            ]
            if not valid_items:
                continue
            source_attention = valid_items[0]["finetuned_temporal"][0]
        else:
            valid_items = teammate_items
            source_attention = valid_items[0]["finetuned_spatial"][0]

        head_indices = compact_head_indices(source_attention, heads)
        fig = render_compact_team_grid(
            valid_items,
            frame_indices,
            head_indices,
            model_type,
            num_patches_h,
            num_patches_w,
            attention_source,
        )
        output_dir = artifacts_dir / attention_source / "compact"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"sample{sample_idx}.{output_format}"
        fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.02, format=output_format)
        plt.close(fig)
        print(f"  Saved compact: {save_path}")


def load_pretrained_vision_models(model_types=None, device: torch.device | None = None):
    if model_types is None:
        model_types = ["siglip2"]

    device = device or get_device()
    pretrained_models = {}
    for model_type in model_types:
        print(f"Loading pretrained {model_type}...")
        model = load_vision_model_eager(model_type)
        model.eval()
        model.to(device)
        pretrained_models[model_type] = model
    return pretrained_models


def process_experiment_selected(
    experiment_name,
    short_name,
    epoch,
    selected_samples,
    df,
    pretrained_vision_models,
    output_format="svg",
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    heads: str = "mean,all",
    attention_sources: set[str] | None = None,
    checkpoint_name: str | None = None,
    experiment_dir: str | None = None,
    device: torch.device | None = None,
    layout: str = "compact",
    frame_indices: str | None = None,
    checkpoint_label: str | None = None,
):
    device = device or get_device()
    exp_dir = experiment_dir_from_name(experiment_name, experiment_dir)
    cfg = load_experiment_config(exp_dir)
    checkpoint_path = find_checkpoint(Path(cfg.path.ckpt), epoch, checkpoint_name)

    contrastive_model, finetuned_vision_model = load_finetuned_vision_model(
        experiment_name,
        epoch,
        cfg,
        checkpoint_path,
        device,
    )
    video_processor, processor_type = init_video_processor(cfg)
    model_type = cfg.model.encoder.model_type
    pretrained_vision_model = pretrained_vision_models[model_type]
    num_patches_h, num_patches_w = get_patch_dimensions(model_type, finetuned_vision_model)

    if attention_sources is None:
        attention_sources = {"spatial", "temporal"}
    if model_type == "vjepa2":
        attention_sources = {"vjepa2"}

    artifacts_dir = artifact_dir / short_name
    if checkpoint_label:
        artifacts_dir = artifacts_dir / checkpoint_label
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    baseline_temporal_model = None
    if "temporal" in attention_sources and getattr(contrastive_model.video_encoder.video_encoder, "temporal", None) is not None:
        baseline_temporal_model = ContrastiveModel(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
        baseline_temporal_model.eval()
        baseline_temporal_model.to(device)

    if layout == "compact":
        sample_indices = sorted({sample_idx for sample_idx, _ in valid_selected_samples(selected_samples, len(df))})
        for sample_idx in tqdm(sample_indices, desc=f"Rendering compact {short_name}"):
            row = df.row(sample_idx, named=True)
            process_compact_team_sample(
                sample_idx,
                row,
                cfg,
                artifacts_dir,
                video_processor,
                processor_type,
                finetuned_vision_model,
                pretrained_vision_model,
                contrastive_model,
                baseline_temporal_model,
                model_type,
                num_patches_h,
                num_patches_w,
                attention_sources,
                heads,
                output_format,
                frame_indices,
                device,
            )
        return

    for sample_idx, teammate_idx in tqdm(selected_samples, desc=f"Rendering {short_name}"):
        if sample_idx < 0 or sample_idx >= len(df):
            print(f"  Skipping sample{sample_idx}: index is outside filtered dataframe length {len(df)}")
            continue
        row = df.row(sample_idx, named=True)
        agent_id = row[f"teammate_{teammate_idx}_id"]
        if agent_id is None:
            print(f"  Skipping sample{sample_idx}_teammate{teammate_idx}: agent_id is None")
            continue

        video_path = construct_video_path(cfg, row["match_id"], str(agent_id), row["round_num"])
        video_clip_result = load_video_clip(cfg, video_path, row["start_seconds"], row["end_seconds"])
        video_clip = video_clip_result["video"] if isinstance(video_clip_result, dict) else video_clip_result
        video_features = transform_video(video_processor, processor_type, video_clip)
        video_tensor = video_features.unsqueeze(0).to(device)

        with torch.no_grad():
            finetuned_spatial = get_attention_weights(finetuned_vision_model, video_tensor, model_type)
            pretrained_spatial = get_attention_weights(pretrained_vision_model, video_tensor, model_type)
            finetuned_temporal = get_temporal_attention_weights(contrastive_model, video_tensor, model_type)

        pretrained_temporal = None
        if finetuned_temporal is not None and baseline_temporal_model is not None:
            # The backbone has an off-the-shelf baseline, but the temporal module is project-specific.
            # This comparison uses the configured temporal architecture before checkpoint finetuning.
            with torch.no_grad():
                pretrained_temporal = get_temporal_attention_weights(baseline_temporal_model, video_tensor, model_type)

        num_frames = video_tensor.shape[1]
        frame_indices = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]

        if "spatial" in attention_sources:
            for head_index in resolve_head_indices(finetuned_spatial[0], heads):
                fig = render_attention_grid(
                    video_clip,
                    frame_indices,
                    finetuned_spatial,
                    pretrained_spatial,
                    model_type,
                    num_patches_h,
                    num_patches_w,
                    "spatial",
                    head_index,
                )
                output_dir = artifacts_dir / "spatial" / head_label(head_index)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"sample{sample_idx}_teammate{teammate_idx}.{output_format}"
                fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.02, format=output_format)
                plt.close(fig)
                print(f"  Saved: {save_path}")

        if "temporal" in attention_sources and finetuned_temporal is not None and pretrained_temporal is not None:
            for head_index in resolve_head_indices(finetuned_temporal[0], heads):
                fig = render_attention_grid(
                    video_clip,
                    frame_indices,
                    finetuned_temporal,
                    pretrained_temporal,
                    model_type,
                    num_patches_h,
                    num_patches_w,
                    "temporal",
                    head_index,
                    comparison_label="Untrained\nTemporal",
                )
                output_dir = artifacts_dir / "temporal" / head_label(head_index)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"sample{sample_idx}_teammate{teammate_idx}.{output_format}"
                fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.02, format=output_format)
                plt.close(fig)
                print(f"  Saved: {save_path}")


def process_experiment(
    experiment_name,
    short_name,
    epoch,
    sample_indices,
    df,
    pretrained_vision_models,
    **kwargs,
):
    selected_samples = []
    for sample_idx in sample_indices:
        row = df.row(sample_idx, named=True)
        for teammate_idx in range(row["num_alive_teammates"]):
            if row[f"teammate_{teammate_idx}_id"] is not None:
                selected_samples.append((sample_idx, teammate_idx))
                break
    return process_experiment_selected(
        experiment_name,
        short_name,
        epoch,
        selected_samples,
        df,
        pretrained_vision_models,
        **kwargs,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate spatial and temporal attention visualizations from contrastive checkpoints."
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--epoch", type=int, default=39)
    parser.add_argument("--epochs", nargs="*", default=None, help="Epoch numbers and/or 'last', e.g. --epochs 1 5 20 last")
    parser.add_argument("--use-last", action="store_true", help="Use checkpoint/last.ckpt instead of --epoch")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output-format", default="svg", choices=["svg", "pdf", "png"])
    parser.add_argument("--heads", default="mean,all", help="mean, all, mean,all, or comma-separated head indices")
    parser.add_argument("--attention-source", default="both", choices=["spatial", "temporal", "both"])
    parser.add_argument("--layout", default="compact", choices=["compact", "per-teammate"])
    parser.add_argument("--frame-indices", default=None, help="Comma-separated frame indices; defaults to all frames")
    parser.add_argument("--selected", action="store_true", help="Use the fixed paper sample/teammate list")
    parser.add_argument("--num-samples", type=int, default=8, help="Random samples when --selected is not passed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    epoch_specs = parse_epoch_specs(args.epochs, args.epoch, args.use_last)
    artifact_dir = Path(args.artifact_dir)

    exp_dir = experiment_dir_from_name(args.experiment, args.experiment_dir)
    cfg = load_experiment_config(exp_dir)
    df = pl.read_csv(cfg.data.label_path, null_values=[])
    df = df.filter(pl.col("partition") == "test")

    if args.selected:
        selected_samples = valid_selected_samples(DEFAULT_SELECTED_SAMPLES, len(df))
    else:
        random.seed(args.seed)
        all_indices = list(range(len(df)))
        sample_indices = sorted(random.sample(all_indices, min(args.num_samples, len(all_indices))))
        selected_samples = []
        for sample_idx in sample_indices:
            row = df.row(sample_idx, named=True)
            teammate_idx = next(
                (
                    idx
                    for idx in range(row["num_alive_teammates"])
                    if row[f"teammate_{idx}_id"] is not None
                ),
                None,
            )
            if teammate_idx is not None:
                selected_samples.append((sample_idx, teammate_idx))

    model_type = cfg.model.encoder.model_type
    sources = {"spatial", "temporal"} if args.attention_source == "both" else {args.attention_source}
    pretrained_vision_models = load_pretrained_vision_models([model_type], device)

    short_name = f"{model_type}-{cfg.data.map}-{cfg.data.ui_mask}"
    print(f"Experiment: {exp_dir.name}")
    print(f"Output: {artifact_dir / short_name}")
    print(f"Device: {device}")
    print(f"Samples: {len(selected_samples)}")
    print(f"Checkpoints: {', '.join(label for label, _ in epoch_specs)}")

    for checkpoint_label, epoch in epoch_specs:
        process_experiment_selected(
            exp_dir.name,
            short_name,
            epoch,
            selected_samples,
            df,
            pretrained_vision_models,
            output_format=args.output_format,
            artifact_dir=artifact_dir,
            heads=args.heads,
            attention_sources=sources,
            checkpoint_name=args.checkpoint_name,
            experiment_dir=str(exp_dir),
            device=device,
            layout=args.layout,
            frame_indices=args.frame_indices,
            checkpoint_label=checkpoint_label,
        )


if __name__ == "__main__":
    main()
