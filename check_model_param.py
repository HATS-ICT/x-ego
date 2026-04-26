import argparse
import gc
from types import SimpleNamespace

import torch

from src.models.modules.video_encoder import MODEL_TYPE_TO_PRETRAINED, VideoEncoder


DEFAULT_MODELS = ("clip", "siglip2", "dinov3", "vjepa2", "resnet50")


def count_parameters(model: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def format_millions(value: int) -> str:
    return f"{value / 1_000_000:.2f}M"


def build_cfg(
    model_type: str,
    finetune_last_k_layers: int,
    temporal_heads: int | None,
    temporal_depth: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_type=model_type,
        finetune_last_k_layers=finetune_last_k_layers,
        temporal_heads=temporal_heads,
        temporal_depth=temporal_depth,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print total, trainable, and non-trainable parameter counts for supported encoders."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=tuple(MODEL_TYPE_TO_PRETRAINED.keys()),
        help="Model aliases to check.",
    )
    parser.add_argument(
        "--finetune-last-k-layers",
        type=int,
        default=3,
        help="Freeze policy used by the project encoder. Use -1 to train all, 0 to freeze all.",
    )
    parser.add_argument(
        "--temporal-heads",
        type=int,
        default=8,
        help="Temporal transformer heads for image encoders and ResNet-50. Use 0 to disable.",
    )
    parser.add_argument(
        "--temporal-depth",
        type=int,
        default=1,
        help="Number of temporal transformer blocks when temporal heads are enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temporal_heads = args.temporal_heads if args.temporal_heads > 0 else None

    rows = []
    for model_type in args.models:
        full_name = MODEL_TYPE_TO_PRETRAINED[model_type]
        cfg = build_cfg(
            model_type=model_type,
            finetune_last_k_layers=args.finetune_last_k_layers,
            temporal_heads=temporal_heads,
            temporal_depth=args.temporal_depth,
        )

        try:
            model = VideoEncoder(cfg)
            total, trainable, non_trainable = count_parameters(model)
            rows.append(
                (
                    model_type,
                    full_name,
                    format_millions(total),
                    format_millions(trainable),
                    format_millions(non_trainable),
                )
            )
            del model
            gc.collect()
        except Exception as exc:
            rows.append((model_type, full_name, "error", "error", str(exc)))

    headers = ("model", "full name", "total", "trainable", "non trainable")
    widths = [
        max(len(str(row[index])) for row in (headers, *rows))
        for index in range(len(headers))
    ]

    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


if __name__ == "__main__":
    main()
