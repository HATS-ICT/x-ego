"""
Analyze how cross-ego contrastive training reshapes SigLIP2 embeddings.

The script has two phases:
1. Cache embeddings for baseline SigLIP2 and selected contrastive checkpoints.
2. Generate alignment metrics, t-SNE plots, cluster summaries, and optional
   language-anchor drift probes.

It is intentionally conservative about dataset size. Start with a few hundred
observations, inspect the plots, then scale up if the result is useful.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_module.contrastive import ContrastiveDataModule
from src.models.contrastive_model import ContrastiveModel
from src.scripts.language_visualization.concept_vocabulary import (
    ALL_CONCEPTS,
    CONCEPT_TO_GROUP,
    GROUP_COLORS,
)
from src.scripts.language_visualization.language_utils import (
    get_text_embeddings,
    load_siglip2_model,
)
from src.utils.env_utils import get_data_base_path, get_output_base_path


if platform.system() == "Windows":
    import pathlib
    import pathlib._local

    pathlib._local.PosixPath = pathlib._local.WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath


DEFAULT_EXPERIMENT = "main_contra_with_accu-siglip2-mirage-ui-all-260427-080317-eixo"
GROUP_ORDER = ["egocentric", "teammate", "enemy", "global", "spatial"]


@dataclass(frozen=True)
class EmbeddingCache:
    tag: str
    epoch: int | None
    npz_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class LanguageVocab:
    name: str
    concepts: list[str]
    groups: list[str]
    counts: list[int | None]
    group_colors: dict[str, str]
    group_sources: list[str] | None = None
    group_scores: list[float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe contrastive embedding geometry and language-anchor drift."
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Optional direct path to an experiment folder containing hparam.yaml.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional direct path to the downloaded per-epoch checkpoint folder.",
    )
    parser.add_argument("--epochs", nargs="*", type=int, default=[1, 4, 10, 19, 27, 39])
    parser.add_argument(
        "--all-epochs",
        action="store_true",
        help="Discover every available epoch checkpoint instead of using --epochs.",
    )
    parser.add_argument(
        "--tsne-epochs",
        nargs="*",
        type=int,
        default=None,
        help="Optional epoch subset for t-SNE grids. Metrics still use all selected epochs.",
    )
    parser.add_argument("--partition", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-observations", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--spaces", nargs="+", default=["vision", "projected"], choices=["vision", "projected"])
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--language-only", action="store_true")
    parser.add_argument("--run-language", action="store_true")
    parser.add_argument("--language-top-k", type=int, default=25)
    parser.add_argument("--language-vocab", default=None, help="Optional JSON vocabulary, e.g. data/wiki/concepts.json")
    parser.add_argument("--language-text-key", default="concept")
    parser.add_argument("--language-count-key", default="count")
    parser.add_argument("--language-max-concepts", type=int, default=0)
    parser.add_argument(
        "--language-prompt-mode",
        choices=["raw", "prompted", "ensemble"],
        default="ensemble",
        help="How to turn a concept into SigLIP2 text input.",
    )
    parser.add_argument(
        "--language-group-mode",
        choices=["manual", "wiki", "tactical"],
        default="wiki",
        help="Grouping for external vocabularies. tactical uses keywords plus SigLIP2 text-anchor similarity.",
    )
    parser.add_argument("--language-keyword-weight", type=float, default=0.08)
    parser.add_argument("--language-group-threshold", type=float, default=0.16)
    parser.add_argument("--language-group-margin", type=float, default=0.02)
    parser.add_argument("--artifact-dir", default=None)
    return parser.parse_args()


def existing_path(path: str | Path | None, description: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"{description} is not configured")
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def get_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    output_base = existing_path(get_output_base_path(), "OUTPUT_BASE_PATH")
    data_base = existing_path(get_data_base_path(), "DATA_BASE_PATH")
    exp_dir = Path(args.experiment_dir).expanduser() if args.experiment_dir else output_base / args.experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    args.experiment = exp_dir.name

    artifact_root = (
        Path(args.artifact_dir)
        if args.artifact_dir
        else Path("artifacts") / "contra_cluster_v2" / args.experiment
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    return output_base, data_base, exp_dir, artifact_root


def get_checkpoint_dir(args: argparse.Namespace, exp_dir: Path) -> Path:
    ckpt_dir = Path(args.checkpoint_dir).expanduser() if args.checkpoint_dir else exp_dir / "checkpoint"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    return ckpt_dir


def discover_checkpoint_epochs(ckpt_dir: Path) -> list[int]:
    epochs = set()
    for path in ckpt_dir.glob("*.ckpt"):
        match = re.search(r"-e(\d+)-", path.name)
        if match:
            epochs.add(int(match.group(1)))
    if not epochs:
        raise FileNotFoundError(f"No per-epoch checkpoints found in {ckpt_dir}")
    return sorted(epochs)


def load_cfg(args: argparse.Namespace, exp_dir: Path, data_base: Path) -> OmegaConf:
    hparam_path = exp_dir / "hparam.yaml"
    if not hparam_path.exists():
        raise FileNotFoundError(f"hparam.yaml not found: {hparam_path}")

    cfg = OmegaConf.load(hparam_path)
    cfg.data.partition = args.partition
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers
    cfg.data.persistent_workers = False
    cfg.data.pin_mem = args.device.startswith("cuda")
    cfg.data.prefetch_factor = None
    cfg.data.random_mask.enable = False

    path_cfg = OmegaConf.create(
        {
            "path": {
                "src": str(Path(__file__).resolve().parents[3]),
                "data": str(data_base),
                "output": str(exp_dir.parent),
                "exp": str(exp_dir),
                "ckpt": str(get_checkpoint_dir(args, exp_dir)),
                "plots": str(exp_dir / "plots"),
            }
        }
    )
    return OmegaConf.merge(cfg, path_cfg)


def find_checkpoint(ckpt_dir: Path, epoch: int) -> Path:
    matches = sorted(ckpt_dir.glob(f"*-e{epoch:02d}-*.ckpt"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint for epoch {epoch} in {ckpt_dir}")
    return matches[0]


def load_model(cfg: OmegaConf, exp_dir: Path, epoch: int | None, device: torch.device) -> ContrastiveModel:
    model = ContrastiveModel(cfg)
    if epoch is not None:
        ckpt_path = find_checkpoint(Path(cfg.path.ckpt), epoch)
        print(f"Loading checkpoint: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ContrastiveModel._strip_orig_mod_prefix(checkpoint["state_dict"])
        model.load_state_dict(state_dict)
    else:
        print("Loading baseline model: off-the-shelf SigLIP2 vision encoder with untrained projector")

    model.to(device)
    model.eval()
    return model


def create_dataloader(cfg: OmegaConf, partition: str):
    dm = ContrastiveDataModule(cfg)
    if partition == "test":
        dm.prepare_data()
        dm.setup("test")
        return dm.test_dataloader()

    dm.prepare_data()
    dm.setup("fit")
    if partition == "train":
        return dm.train_dataloader()
    if partition == "val":
        return dm.val_dataloader()

    raise ValueError(f"Unsupported partition: {partition}")


def cache_tag(epoch: int | None) -> str:
    return "baseline" if epoch is None else f"epoch_{epoch:02d}"


def cache_paths(artifact_root: Path, epoch: int | None) -> EmbeddingCache:
    tag = cache_tag(epoch)
    return EmbeddingCache(
        tag=tag,
        epoch=epoch,
        npz_path=artifact_root / "cache" / f"{tag}_embeddings.npz",
        metadata_path=artifact_root / "cache" / f"{tag}_metadata.csv",
    )


def write_metadata(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metadata rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def expand_metadata(batch: dict, seen_observations: int, seen_agents: int) -> list[dict]:
    rows = []
    agent_counts = batch["agent_counts"].tolist()
    sample_sides = batch["pov_team_side"]
    sample_indices = batch["original_csv_idx"]
    flat_agent_ids = [str(a) for a in batch["agent_ids"]]

    agent_offset = 0
    for local_obs_idx, (csv_idx, count) in enumerate(zip(sample_indices, agent_counts)):
        for agent_rank in range(count):
            rows.append(
                {
                    "row_id": seen_agents + len(rows),
                    "observation_order": seen_observations + local_obs_idx,
                    "csv_idx": int(csv_idx),
                    "agent_id": flat_agent_ids[agent_offset + agent_rank],
                    "agent_rank": agent_rank,
                    "num_agents": int(count),
                    "pov_team_side": str(sample_sides[local_obs_idx]).lower(),
                }
            )
        agent_offset += count
    return rows


def resolve_label_path(cfg: OmegaConf) -> Path:
    data_root = Path(cfg.path.data)
    candidates = []
    if "map" in cfg.data:
        candidates.append(data_root / cfg.data.map / cfg.data.labels_folder / cfg.data.labels_filename)
    candidates.append(data_root / cfg.data.labels_folder / cfg.data.labels_filename)

    for path in candidates:
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find labels file. Searched: {searched}")


def enrich_metadata(metadata_path: Path, cfg: OmegaConf) -> pl.DataFrame:
    metadata = pl.read_csv(metadata_path)
    label_path = resolve_label_path(cfg)
    labels = pl.read_csv(label_path, null_values=[]).rename({"idx": "csv_idx"})
    join_columns = [
        "csv_idx",
        "partition",
        "start_seconds",
        "end_seconds",
        "match_id",
        "round_num",
        "num_alive_teammates",
    ]
    optional_columns = [c for c in ["start_tick", "end_tick"] if c in labels.columns]
    labels = labels.select(join_columns + optional_columns)
    return metadata.join(labels, on="csv_idx", how="left")


def cache_embeddings(
    args: argparse.Namespace,
    cfg: OmegaConf,
    exp_dir: Path,
    artifact_root: Path,
    epoch: int | None,
) -> EmbeddingCache:
    cache = cache_paths(artifact_root, epoch)
    if cache.npz_path.exists() and cache.metadata_path.exists() and not args.force_cache:
        print(f"Reusing cache: {cache.npz_path}")
        return cache

    device = torch.device(args.device)
    model = load_model(cfg, exp_dir, epoch, device)
    loader = create_dataloader(cfg, args.partition)

    all_vision = []
    all_projected = []
    metadata_rows = []
    seen_observations = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Embedding {cache.tag}"):
            if args.max_observations and seen_observations >= args.max_observations:
                break

            observation_count = len(batch["agent_counts"])
            if args.max_observations:
                keep_observations = min(observation_count, args.max_observations - seen_observations)
                if keep_observations <= 0:
                    break
                keep_agents = int(batch["agent_counts"][:keep_observations].sum().item())
                batch = {
                    **batch,
                    "video": batch["video"][:keep_agents],
                    "agent_counts": batch["agent_counts"][:keep_observations],
                    "agent_ids": batch["agent_ids"][:keep_agents],
                    "pov_team_side": batch["pov_team_side"][:keep_observations],
                    "original_csv_idx": batch["original_csv_idx"][:keep_observations],
                }
                observation_count = keep_observations

            video = batch["video"].to(device, non_blocking=True)
            vision = model.video_encoder(video)
            projected = model.video_projector(vision)

            all_vision.append(vision.detach().float().cpu())
            all_projected.append(projected.detach().float().cpu())
            metadata_rows.extend(expand_metadata(batch, seen_observations, len(metadata_rows)))
            seen_observations += observation_count

    if not all_vision:
        raise RuntimeError("No embeddings were produced")

    vision_np = torch.cat(all_vision, dim=0).numpy()
    projected_np = torch.cat(all_projected, dim=0).numpy()

    cache.npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache.npz_path, vision=vision_np, projected=projected_np)
    write_metadata(cache.metadata_path, metadata_rows)
    enrich_metadata(cache.metadata_path, cfg).write_csv(cache.metadata_path)

    print(f"Saved {len(vision_np)} embeddings to {cache.npz_path}")
    return cache


def load_embeddings(cache: EmbeddingCache, space: str) -> tuple[np.ndarray, pl.DataFrame]:
    data = np.load(cache.npz_path)
    embeddings = data[space].astype(np.float32)
    metadata = pl.read_csv(cache.metadata_path)
    if len(metadata) != len(embeddings):
        raise ValueError(f"Metadata/embedding length mismatch for {cache.tag}")
    return embeddings, metadata


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


def pairwise_cosine(x: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(x)
    return normalized @ normalized.T


def safe_silhouette(x: np.ndarray, labels: Iterable, seed: int) -> float:
    labels = np.asarray(list(labels))
    valid = np.array([str(v) not in {"", "None", "nan", "null"} for v in labels])
    if valid.sum() < 4:
        return float("nan")
    labels = labels[valid]
    x = x[valid]
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return float("nan")
    if len(x) > 1500:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=1500, replace=False)
        x = x[idx]
        labels = labels[idx]
    return float(silhouette_score(l2_normalize(x), labels, metric="cosine"))


def compute_alignment_metrics(
    cache: EmbeddingCache,
    space: str,
    embeddings: np.ndarray,
    metadata: pl.DataFrame,
    seed: int,
) -> dict:
    sim = pairwise_cosine(embeddings)
    obs = metadata["csv_idx"].to_numpy()
    same = obs[:, None] == obs[None, :]
    diag = np.eye(len(obs), dtype=bool)
    pos_mask = same & ~diag
    neg_mask = ~same

    pos_mean = float(sim[pos_mask].mean()) if pos_mask.any() else float("nan")
    neg_mean = float(sim[neg_mask].mean()) if neg_mask.any() else float("nan")

    masked = sim.copy()
    masked[diag] = -np.inf
    nearest = np.argmax(masked, axis=1)
    top1_same_obs = float(np.mean(obs[nearest] == obs))

    k = min(5, len(obs) - 1)
    if k > 0:
        topk = np.argpartition(masked, -k, axis=1)[:, -k:]
        top5_same_obs = float(np.mean([(obs[row] == obs[topk[row]]).any() for row in range(len(obs))]))
    else:
        top5_same_obs = float("nan")

    team_labels = metadata["pov_team_side"].to_list() if "pov_team_side" in metadata.columns else []
    alive_labels = metadata["num_alive_teammates"].to_list() if "num_alive_teammates" in metadata.columns else []

    return {
        "tag": cache.tag,
        "epoch": -1 if cache.epoch is None else cache.epoch,
        "space": space,
        "n_agents": len(embeddings),
        "n_observations": int(len(np.unique(obs))),
        "same_observation_cosine": pos_mean,
        "different_observation_cosine": neg_mean,
        "positive_margin": pos_mean - neg_mean,
        "same_observation_top1": top1_same_obs,
        "same_observation_top5": top5_same_obs,
        "team_side_silhouette": safe_silhouette(embeddings, team_labels, seed),
        "alive_count_silhouette": safe_silhouette(embeddings, alive_labels, seed),
    }


def cache_label(cache: EmbeddingCache) -> str:
    return "base" if cache.epoch is None else str(cache.epoch)


def ordered_caches(caches: list[EmbeddingCache]) -> list[EmbeddingCache]:
    return sorted(caches, key=lambda c: -1 if c.epoch is None else c.epoch)


def select_plot_caches(caches: list[EmbeddingCache], tsne_epochs: list[int] | None) -> list[EmbeddingCache]:
    ordered = ordered_caches(caches)
    if tsne_epochs is None:
        return ordered

    requested = set(tsne_epochs)
    selected = [cache for cache in ordered if cache.epoch is None or cache.epoch in requested]
    if len(selected) < 2:
        raise ValueError("--tsne-epochs must leave at least one epoch plus the baseline for plotting")
    return selected


def plot_alignment_trajectories(artifact_root: Path, metrics: list[dict]) -> None:
    df = pl.DataFrame(metrics).sort(["space", "epoch"])
    metrics_to_plot = [
        ("positive_margin", "Positive minus negative cosine"),
        ("same_observation_cosine", "Same-observation cosine"),
        ("different_observation_cosine", "Different-observation cosine"),
        ("same_observation_top5", "Top-5 same-observation retrieval"),
        ("team_side_silhouette", "Team-side silhouette"),
        ("alive_count_silhouette", "Alive-count silhouette"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics_to_plot):
        for space in sorted(df["space"].unique().to_list()):
            sub = df.filter(pl.col("space") == space).sort("epoch")
            x = sub["epoch"].to_numpy()
            y = sub[metric].to_numpy()
            labels = ["base" if int(epoch) < 0 else str(int(epoch)) for epoch in x]
            ax.plot(np.arange(len(x)), y, marker="o", linewidth=1.8, label=space)
            ax.set_xticks(np.arange(len(x)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="best")
    fig.suptitle("Contrastive alignment trajectory across checkpoints", fontsize=14)
    fig.tight_layout()
    out = artifact_root / "alignment_trajectory.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def metadata_keys(metadata: pl.DataFrame) -> list[str]:
    required = ["csv_idx", "agent_id", "agent_rank"]
    missing = [col for col in required if col not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata missing key columns: {missing}")

    return [
        f"{csv_idx}::{agent_id}::{agent_rank}"
        for csv_idx, agent_id, agent_rank in zip(
            metadata["csv_idx"].to_list(),
            metadata["agent_id"].to_list(),
            metadata["agent_rank"].to_list(),
        )
    ]


def compute_embedding_change_metrics(caches: list[EmbeddingCache], spaces: list[str]) -> list[dict]:
    ordered = ordered_caches(caches)
    baseline = ordered[0]
    if baseline.epoch is not None:
        raise ValueError("Embedding drift metrics require a baseline cache")

    rows = []
    for space in spaces:
        baseline_embeddings, baseline_metadata = load_embeddings(baseline, space)
        baseline_keys = metadata_keys(baseline_metadata)
        baseline_index = {key: idx for idx, key in enumerate(baseline_keys)}

        for cache in ordered:
            embeddings, metadata = load_embeddings(cache, space)
            current_keys = metadata_keys(metadata)
            current_index = {key: idx for idx, key in enumerate(current_keys)}
            common_keys = [key for key in baseline_keys if key in current_index]
            if not common_keys:
                continue

            base_idx = np.array([baseline_index[key] for key in common_keys], dtype=int)
            cur_idx = np.array([current_index[key] for key in common_keys], dtype=int)
            base = baseline_embeddings[base_idx]
            current = embeddings[cur_idx]
            base_norm = l2_normalize(base)
            current_norm = l2_normalize(current)
            cosine_to_baseline = np.sum(base_norm * current_norm, axis=1)
            delta_norm = np.linalg.norm(current_norm - base_norm, axis=1)

            rows.append(
                {
                    "tag": cache.tag,
                    "epoch": -1 if cache.epoch is None else cache.epoch,
                    "space": space,
                    "n_matched_agents": len(common_keys),
                    "mean_cosine_to_baseline": float(cosine_to_baseline.mean()),
                    "median_cosine_to_baseline": float(np.median(cosine_to_baseline)),
                    "mean_unit_delta_norm": float(delta_norm.mean()),
                    "p95_unit_delta_norm": float(np.percentile(delta_norm, 95)),
                }
            )
    return rows


def plot_embedding_change_trajectories(artifact_root: Path, rows: list[dict]) -> None:
    if not rows:
        return

    df = pl.DataFrame(rows).sort(["space", "epoch"])
    metrics_to_plot = [
        ("mean_cosine_to_baseline", "Mean cosine to baseline"),
        ("median_cosine_to_baseline", "Median cosine to baseline"),
        ("mean_unit_delta_norm", "Mean unit-vector delta"),
        ("p95_unit_delta_norm", "P95 unit-vector delta"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    axes = axes.ravel()

    for ax, (metric, title) in zip(axes, metrics_to_plot):
        for space in sorted(df["space"].unique().to_list()):
            sub = df.filter(pl.col("space") == space).sort("epoch")
            x = sub["epoch"].to_numpy()
            y = sub[metric].to_numpy()
            labels = ["base" if int(epoch) < 0 else str(int(epoch)) for epoch in x]
            ax.plot(np.arange(len(x)), y, marker="o", linewidth=1.8, label=space)
            ax.set_xticks(np.arange(len(x)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="best")
    fig.suptitle("How far each paired agent embedding moved from baseline", fontsize=14)
    fig.tight_layout()
    out = artifact_root / "embedding_change_trajectory.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def compute_tsne_points(
    embeddings_by_tag: dict[str, np.ndarray],
    perplexity: float,
    seed: int,
) -> dict[str, np.ndarray]:
    tags = list(embeddings_by_tag)
    arrays = [l2_normalize(embeddings_by_tag[tag]) for tag in tags]
    all_embeddings = np.concatenate(arrays, axis=0)

    n_components = min(50, all_embeddings.shape[1], max(2, all_embeddings.shape[0] - 1))
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(all_embeddings)
    actual_perplexity = min(perplexity, max(1.0, (len(all_embeddings) - 1) / 3), len(all_embeddings) - 1.0)
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=actual_perplexity,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(reduced)

    result = {}
    start = 0
    for tag, arr in zip(tags, arrays):
        result[tag] = coords[start : start + len(arr)]
        start += len(arr)
    return result


def plot_tsne_grid(
    artifact_root: Path,
    space: str,
    caches: list[EmbeddingCache],
    coords_by_tag: dict[str, np.ndarray],
    metadata_by_tag: dict[str, pl.DataFrame],
    color_key: str,
) -> None:
    n = len(caches)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), squeeze=False)
    axes = axes[0]

    for ax, cache in zip(axes, caches):
        coords = coords_by_tag[cache.tag]
        metadata = metadata_by_tag[cache.tag]

        if color_key == "team_side":
            values = metadata["pov_team_side"].to_list()
            color_map = {"t": "#d95f02", "ct": "#1b9e77"}
            colors = [color_map.get(str(v).lower(), "#7570b3") for v in values]
            ax.scatter(coords[:, 0], coords[:, 1], s=5, c=colors, alpha=0.68, linewidths=0)
        elif color_key == "alive_count":
            values = metadata["num_alive_teammates"].to_numpy()
            sc = ax.scatter(coords[:, 0], coords[:, 1], s=5, c=values, cmap="viridis", alpha=0.68, linewidths=0)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        elif color_key == "start_seconds":
            values = metadata["start_seconds"].to_numpy()
            sc = ax.scatter(coords[:, 0], coords[:, 1], s=5, c=values, cmap="magma", alpha=0.68, linewidths=0)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        elif color_key == "cluster":
            cluster_col = f"{space}_cluster"
            values = metadata[cluster_col].to_numpy()
            ax.scatter(coords[:, 0], coords[:, 1], s=5, c=values, cmap="tab20", alpha=0.72, linewidths=0)
        else:
            raise ValueError(f"Unsupported color key: {color_key}")

        ax.set_title(cache.tag.replace("_", " "))
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{space} t-SNE colored by {color_key}", fontsize=14)
    fig.tight_layout()
    out = artifact_root / f"tsne_{space}_{color_key}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def add_clusters_and_summarize(
    artifact_root: Path,
    cache: EmbeddingCache,
    space: str,
    embeddings: np.ndarray,
    metadata: pl.DataFrame,
    n_clusters: int,
    seed: int,
) -> pl.DataFrame:
    n_clusters = min(n_clusters, max(2, len(embeddings) // 10))
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(l2_normalize(embeddings))
    metadata = metadata.with_columns(pl.Series(f"{space}_cluster", labels))

    summary = (
        metadata.group_by(f"{space}_cluster")
        .agg(
            [
                pl.len().alias("n_agents"),
                pl.col("csv_idx").n_unique().alias("n_observations"),
                pl.col("pov_team_side").mode().first().alias("majority_team_side"),
                pl.col("num_alive_teammates").mean().alias("mean_alive_teammates"),
                pl.col("start_seconds").mean().alias("mean_start_seconds"),
                pl.col("round_num").mode().first().alias("mode_round_num"),
                pl.col("match_id").mode().first().alias("mode_match_id"),
            ]
        )
        .sort("n_agents", descending=True)
    )

    out = artifact_root / f"cluster_summary_{space}_{cache.tag}.csv"
    summary.write_csv(out)
    return metadata


def write_alignment_and_plots(
    args: argparse.Namespace,
    cfg: OmegaConf,
    artifact_root: Path,
    caches: list[EmbeddingCache],
) -> None:
    caches = ordered_caches(caches)
    tsne_caches = select_plot_caches(caches, args.tsne_epochs)
    metrics = []

    for space in args.spaces:
        embeddings_by_tag = {}
        metadata_by_tag = {}

        for cache in caches:
            embeddings, metadata = load_embeddings(cache, space)
            metrics.append(compute_alignment_metrics(cache, space, embeddings, metadata, args.seed))
            metadata = add_clusters_and_summarize(
                artifact_root, cache, space, embeddings, metadata, args.n_clusters, args.seed
            )
            if cache in tsne_caches:
                embeddings_by_tag[cache.tag] = embeddings
                metadata_by_tag[cache.tag] = metadata

        coords_by_tag = compute_tsne_points(embeddings_by_tag, args.perplexity, args.seed)
        for color_key in ["team_side", "alive_count", "start_seconds", "cluster"]:
            plot_tsne_grid(artifact_root, space, tsne_caches, coords_by_tag, metadata_by_tag, color_key)

    metrics_path = artifact_root / "alignment_metrics.csv"
    pl.DataFrame(metrics).sort(["space", "epoch"]).write_csv(metrics_path)
    print(f"Saved {metrics_path}")
    plot_alignment_trajectories(artifact_root, metrics)

    change_rows = compute_embedding_change_metrics(caches, args.spaces)
    change_path = artifact_root / "embedding_change_metrics.csv"
    pl.DataFrame(change_rows).sort(["space", "epoch"]).write_csv(change_path)
    print(f"Saved {change_path}")
    plot_embedding_change_trajectories(artifact_root, change_rows)


WIKI_GROUP_COLORS = {
    "weapon": "#d95f02",
    "equipment": "#7570b3",
    "mechanic": "#1b9e77",
    "team_player": "#e7298a",
    "objective_mode": "#66a61e",
    "franchise": "#666666",
    "other": "#a6761d",
}

TACTICAL_GROUP_COLORS = {
    "egocentric": "#e74c3c",
    "team": "#2ecc71",
    "enemy": "#3498db",
    "global": "#9b59b6",
    "spatial": "#95a5a6",
    "other": "#7f8c8d",
}

TACTICAL_GROUP_KEYWORDS = {
    "egocentric": [
        "player",
        "weapon",
        "gun",
        "pistol",
        "rifle",
        "sniper",
        "shotgun",
        "knife",
        "fire",
        "firing",
        "reload",
        "reloading",
        "magazine",
        "ammo",
        "ammunition",
        "recoil",
        "accuracy",
        "crosshair",
        "damage",
        "health",
        "armor",
        "kevlar",
        "helmet",
        "movement",
        "speed",
        "view",
        "scope",
    ],
    "team": [
        "teammate",
        "teammates",
        "ally",
        "allies",
        "friendly",
        "team",
        "cooperative",
        "assist",
        "support",
        "trade",
        "squad",
    ],
    "enemy": [
        "enemy",
        "enemies",
        "opponent",
        "opponents",
        "hostile",
        "victim",
        "terrorist",
        "terrorists",
        "counter-terrorist",
        "counter-terrorists",
        "ct",
        "t-side",
    ],
    "global": [
        "round",
        "match",
        "score",
        "objective",
        "bomb",
        "plant",
        "planted",
        "defuse",
        "defusal",
        "hostage",
        "economy",
        "money",
        "cash",
        "award",
        "kill award",
        "competitive",
        "casual",
        "deathmatch",
        "wingman",
        "mode",
        "game",
    ],
    "spatial": [
        "map",
        "site",
        "bombsite",
        "spawn",
        "middle",
        "mid",
        "long",
        "short",
        "connector",
        "ramp",
        "palace",
        "window",
        "jungle",
        "catwalk",
        "tunnel",
        "ladder",
        "door",
        "room",
        "area",
        "position",
        "location",
        "cover",
    ],
}

TACTICAL_GROUP_ANCHORS = {
    "egocentric": [
        "first person player perspective",
        "the player holding and firing a weapon",
        "player health armor ammo recoil accuracy and movement",
        "what the player sees and controls",
    ],
    "team": [
        "teammates allies and friendly players",
        "team coordination support trading and covering",
        "friendly team strategy and teammate positions",
    ],
    "enemy": [
        "enemy team opponents and hostile players",
        "enemy threat enemy position and enemy actions",
        "opponents attacking shooting or being killed",
    ],
    "global": [
        "round state bomb objective economy score and game mode",
        "bomb plant defuse hostage match outcome and kill award",
        "global game rules objectives and round progress",
    ],
    "spatial": [
        "map locations bombsites areas and positions",
        "spatial callouts connector ramp palace window spawn",
        "where players are located on the map",
    ],
}

WEAPON_KEYWORDS = {
    "ak-47",
    "m4a4",
    "m4a1-s",
    "awp",
    "ssg 08",
    "glock-18",
    "usp-s",
    "p2000",
    "desert eagle",
    "five-seven",
    "dual berettas",
    "p250",
    "tec-9",
    "cz75-auto",
    "r8 revolver",
    "mac-10",
    "mp9",
    "mp7",
    "mp5-sd",
    "ump-45",
    "p90",
    "pp-bizon",
    "galil ar",
    "famas",
    "aug",
    "sg 553",
    "scar-20",
    "g3sg1",
    "nova",
    "xm1014",
    "mag-7",
    "sawed-off",
    "m249",
    "negev",
    "knife",
    "bayonet",
    "karambit",
}


def infer_wiki_group(concept: str) -> str:
    text = concept.lower()
    if any(keyword in text for keyword in WEAPON_KEYWORDS):
        return "weapon"
    if any(keyword in text for keyword in ["grenade", "flashbang", "smoke", "molotov", "decoy", "kevlar", "armor", "defuse kit", "zeus"]):
        return "equipment"
    if any(keyword in text for keyword in ["damage", "penetration", "accuracy", "recoil", "reload", "magazine", "fire rate", "kill award", "range", "spread"]):
        return "mechanic"
    if any(keyword in text for keyword in ["terrorist", "counter-terrorist", "player", "hostage", "enemy", "team"]):
        return "team_player"
    if any(keyword in text for keyword in ["bomb", "defuse", "plant", "round", "competitive", "casual", "deathmatch", "wingman", "map"]):
        return "objective_mode"
    if any(keyword in text for keyword in ["counter-strike", "global offensive", "valve", "source", "condition zero"]):
        return "franchise"
    return "other"


def load_language_vocab(args: argparse.Namespace) -> LanguageVocab:
    if args.language_vocab is None:
        concepts = list(ALL_CONCEPTS)
        return LanguageVocab(
            name="manual",
            concepts=concepts,
            groups=[CONCEPT_TO_GROUP.get(c, "other") for c in concepts],
            counts=[None for _ in concepts],
            group_colors=dict(GROUP_COLORS),
            group_sources=["manual" for _ in concepts],
            group_scores=[1.0 for _ in concepts],
        )

    vocab_path = Path(args.language_vocab)
    with vocab_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                records.append({"text": item, "count": None, "group": infer_wiki_group(item)})
            elif isinstance(item, dict):
                text = str(item.get(args.language_text_key, "")).strip()
                if text:
                    count = item.get(args.language_count_key)
                    records.append(
                        {
                            "text": text,
                            "count": int(count) if count is not None else None,
                            "group": str(item.get("group") or infer_wiki_group(text)),
                        }
                    )
    elif isinstance(raw, dict):
        for key, value in raw.items():
            count = value if isinstance(value, (int, float)) else None
            records.append({"text": str(key), "count": int(count) if count is not None else None, "group": infer_wiki_group(str(key))})
    else:
        raise ValueError(f"Unsupported vocabulary JSON shape: {vocab_path}")

    seen = set()
    deduped = []
    for record in records:
        norm = record["text"].lower()
        if norm not in seen:
            seen.add(norm)
            deduped.append(record)

    if args.language_max_concepts > 0:
        deduped = deduped[: args.language_max_concepts]
    if not deduped:
        raise ValueError(f"No concepts loaded from {vocab_path}")

    if args.language_group_mode == "tactical":
        groups = ["other" for _ in deduped]
        group_colors = dict(TACTICAL_GROUP_COLORS)
        group_sources = ["pending" for _ in deduped]
        group_scores = [0.0 for _ in deduped]
    elif args.language_group_mode == "manual" and args.language_vocab is None:
        groups = [CONCEPT_TO_GROUP.get(r["text"], "other") for r in deduped]
        group_colors = dict(GROUP_COLORS)
        group_sources = ["manual" for _ in deduped]
        group_scores = [1.0 for _ in deduped]
    else:
        groups = [r["group"] for r in deduped]
        group_colors = dict(WIKI_GROUP_COLORS)
        group_sources = ["wiki_keyword" for _ in deduped]
        group_scores = [0.0 for _ in deduped]

    return LanguageVocab(
        name=vocab_path.stem,
        concepts=[r["text"] for r in deduped],
        groups=groups,
        counts=[r["count"] for r in deduped],
        group_colors=group_colors,
        group_sources=group_sources,
        group_scores=group_scores,
    )


def concept_groups(vocab: LanguageVocab) -> list[str]:
    preferred = GROUP_ORDER + list(WIKI_GROUP_COLORS)
    present = set(vocab.groups)
    ordered = [group for group in preferred if group in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def make_prompt_variants(concept: str, mode: str) -> list[str]:
    if mode == "raw":
        return [concept]
    if mode == "prompted":
        return [f"a Counter-Strike gameplay screenshot showing {concept}"]
    return [
        concept,
        f"a Counter-Strike gameplay screenshot showing {concept}",
        f"a first person Counter-Strike scene with {concept}",
        f"gameplay footage of {concept}",
    ]


def get_language_embeddings(
    device: torch.device,
    vocab: LanguageVocab,
    prompt_mode: str,
) -> tuple[torch.nn.Module, object, torch.Tensor]:
    model, processor = load_siglip2_model()
    model.to(device)
    model.eval()

    flat_texts = []
    spans = []
    for concept in vocab.concepts:
        variants = make_prompt_variants(concept, prompt_mode)
        start = len(flat_texts)
        flat_texts.extend(variants)
        spans.append((start, len(flat_texts)))

    flat_embeds = []
    chunk_size = 256
    for start in tqdm(range(0, len(flat_texts), chunk_size), desc=f"Text embeddings ({vocab.name})"):
        chunk = flat_texts[start : start + chunk_size]
        flat_embeds.append(get_text_embeddings(model, processor, chunk, device).cpu())
    flat_embeds = torch.cat(flat_embeds, dim=0)

    concept_embeds = []
    for start, end in spans:
        embeds = flat_embeds[start:end]
        pooled = F.normalize(embeds.mean(dim=0, keepdim=True), p=2, dim=-1)
        concept_embeds.append(pooled)
    text_embeds = torch.cat(concept_embeds, dim=0).to(device)
    return model, processor, text_embeds


def keyword_group_scores(text: str) -> dict[str, float]:
    lowered = text.lower()
    scores = {}
    for group, keywords in TACTICAL_GROUP_KEYWORDS.items():
        hits = 0
        for keyword in keywords:
            if keyword in lowered:
                hits += 1
        scores[group] = float(hits)
    return scores


def classify_tactical_groups(
    args: argparse.Namespace,
    vocab: LanguageVocab,
    model: torch.nn.Module,
    processor: object,
    device: torch.device,
    text_np: np.ndarray,
) -> LanguageVocab:
    anchor_groups = list(TACTICAL_GROUP_ANCHORS)
    anchor_embeds = []
    for group in anchor_groups:
        embeds = get_text_embeddings(model, processor, TACTICAL_GROUP_ANCHORS[group], device)
        pooled = F.normalize(embeds.mean(dim=0, keepdim=True), p=2, dim=-1)
        anchor_embeds.append(pooled.cpu().numpy()[0])
    anchor_np = l2_normalize(np.stack(anchor_embeds, axis=0).astype(np.float32))

    semantic_scores = text_np @ anchor_np.T
    groups = []
    sources = []
    final_scores = []

    for idx, concept in enumerate(vocab.concepts):
        kw_scores = keyword_group_scores(concept)
        combined = {}
        for group_idx, group in enumerate(anchor_groups):
            combined[group] = float(semantic_scores[idx, group_idx]) + args.language_keyword_weight * kw_scores[group]

        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        best_group, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
        margin = best_score - second_score
        best_keyword_group = max(kw_scores, key=kw_scores.get)
        best_keyword_hits = kw_scores[best_keyword_group]

        if best_keyword_hits > 0 and combined[best_keyword_group] >= best_score - args.language_keyword_weight:
            assigned = best_keyword_group
            source = "keyword+similarity"
            score = combined[assigned]
        elif best_score >= args.language_group_threshold and margin >= args.language_group_margin:
            assigned = best_group
            source = "similarity"
            score = best_score
        else:
            assigned = "other"
            source = "below_threshold"
            score = best_score

        groups.append(assigned)
        sources.append(source)
        final_scores.append(float(score))

    return LanguageVocab(
        name=vocab.name,
        concepts=vocab.concepts,
        groups=groups,
        counts=vocab.counts,
        group_colors=dict(TACTICAL_GROUP_COLORS),
        group_sources=sources,
        group_scores=final_scores,
    )


def compute_language_metrics(
    args: argparse.Namespace,
    artifact_root: Path,
    caches: list[EmbeddingCache],
) -> None:
    vocab = load_language_vocab(args)
    language_root = artifact_root if vocab.name == "manual" else artifact_root / f"language_{vocab.name}"
    language_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    text_model, text_processor, text_embeds = get_language_embeddings(device, vocab, args.language_prompt_mode)
    text_np = text_embeds.detach().float().cpu().numpy()
    text_np = l2_normalize(text_np)

    if args.language_group_mode == "tactical":
        vocab = classify_tactical_groups(
            args=args,
            vocab=vocab,
            model=text_model,
            processor=text_processor,
            device=device,
            text_np=text_np,
        )

    group_rows = []
    concept_rows = []
    concept_group = np.array(vocab.groups)
    baseline_scores = None
    final_scores = None
    final_tag = None

    for cache in ordered_caches(caches):
        vision, _ = load_embeddings(cache, "vision")
        sims = l2_normalize(vision) @ text_np.T
        mean_scores = sims.mean(axis=0)

        if cache.epoch is None:
            baseline_scores = mean_scores
        final_scores = mean_scores
        final_tag = cache.tag

        for group in concept_groups(vocab):
            mask = concept_group == group
            group_rows.append(
                {
                    "tag": cache.tag,
                    "epoch": -1 if cache.epoch is None else cache.epoch,
                    "group": group,
                    "mean_similarity": float(mean_scores[mask].mean()),
                    "median_similarity": float(np.median(mean_scores[mask])),
                }
            )

        group_sources = vocab.group_sources or ["unknown" for _ in vocab.concepts]
        group_scores = vocab.group_scores or [0.0 for _ in vocab.concepts]
        for concept, group, count, group_source, group_score, score in zip(
            vocab.concepts,
            concept_group,
            vocab.counts,
            group_sources,
            group_scores,
            mean_scores,
        ):
            concept_rows.append(
                {
                    "tag": cache.tag,
                    "epoch": -1 if cache.epoch is None else cache.epoch,
                    "concept": concept,
                    "group": group,
                    "count": count,
                    "group_source": group_source,
                    "group_score": group_score,
                    "mean_similarity": float(score),
                }
            )

    group_df = pl.DataFrame(group_rows).sort(["group", "epoch"])
    group_df.write_csv(language_root / "language_group_metrics.csv")
    pl.DataFrame(concept_rows).sort(["concept", "epoch"]).write_csv(
        language_root / "language_concept_metrics.csv"
    )

    if baseline_scores is not None and final_scores is not None:
        drift = final_scores - baseline_scores
        centered_drift = drift - float(drift.mean())
        order = np.argsort(np.abs(drift))[::-1]
        rows = []
        for idx in order:
            rows.append(
                {
                    "concept": vocab.concepts[idx],
                    "group": concept_group[idx],
                    "count": vocab.counts[idx],
                    "group_source": (vocab.group_sources or ["unknown"] * len(vocab.concepts))[idx],
                    "group_score": (vocab.group_scores or [0.0] * len(vocab.concepts))[idx],
                    "baseline_similarity": float(baseline_scores[idx]),
                    f"{final_tag}_similarity": float(final_scores[idx]),
                    "delta": float(drift[idx]),
                    "centered_delta": float(centered_drift[idx]),
                }
            )
        pl.DataFrame(rows).write_csv(language_root / "language_concept_drift.csv")

    save_language_vocab_manifest(language_root, vocab, args)
    plot_language_group_metrics(language_root, group_df, vocab)
    plot_language_group_delta(language_root, group_df, vocab)
    plot_language_concept_deltas(language_root, concept_rows, vocab)
    plot_language_centered_concept_deltas(language_root, concept_rows, vocab)
    plot_language_delta_distribution(language_root, concept_rows)
    plot_language_baseline_final_scatter(language_root, concept_rows, vocab)
    plot_language_group_boxplot(language_root, concept_rows, vocab)
    plot_language_all_concepts_heatmap(language_root, concept_rows)
    plot_language_rank_stability(language_root, concept_rows)
    plot_language_top_drift_heatmap(language_root, concept_rows, top_k=args.language_top_k)
    plot_language_centered_drift_heatmap(language_root, concept_rows, top_k=args.language_top_k)
    plot_language_anchor_tsne(args, language_root, caches, text_np, concept_group, vocab)


def save_language_vocab_manifest(language_root: Path, vocab: LanguageVocab, args: argparse.Namespace) -> None:
    manifest = {
        "name": vocab.name,
        "n_concepts": len(vocab.concepts),
        "groups": {group: vocab.groups.count(group) for group in concept_groups(vocab)},
        "prompt_mode": args.language_prompt_mode,
        "group_mode": args.language_group_mode,
        "keyword_weight": args.language_keyword_weight,
        "group_threshold": args.language_group_threshold,
        "group_margin": args.language_group_margin,
        "vocab_path": args.language_vocab,
        "text_key": args.language_text_key,
        "count_key": args.language_count_key,
    }
    with (language_root / "language_vocab_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    assignment_rows = []
    group_sources = vocab.group_sources or ["unknown" for _ in vocab.concepts]
    group_scores = vocab.group_scores or [0.0 for _ in vocab.concepts]
    for concept, group, count, source, score in zip(
        vocab.concepts,
        vocab.groups,
        vocab.counts,
        group_sources,
        group_scores,
    ):
        assignment_rows.append(
            {
                "concept": concept,
                "group": group,
                "count": count,
                "group_source": source,
                "group_score": score,
            }
        )
    pl.DataFrame(assignment_rows).write_csv(language_root / "language_vocab_assignments.csv")


def plot_language_group_metrics(artifact_root: Path, group_df: pl.DataFrame, vocab: LanguageVocab) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for group in concept_groups(vocab):
        df = group_df.filter(pl.col("group") == group).sort("epoch")
        x = ["base" if e == -1 else str(e) for e in df["epoch"].to_list()]
        y = df["mean_similarity"].to_list()
        ax.plot(x, y, marker="o", label=group, color=vocab.group_colors.get(group))
    ax.set_title("Mean video-to-language similarity by concept group")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Mean cosine similarity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = artifact_root / "language_group_drift.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_language_group_delta(artifact_root: Path, group_df: pl.DataFrame, vocab: LanguageVocab) -> None:
    epochs = sorted(group_df["epoch"].unique().to_list())
    if -1 not in epochs or len(epochs) < 2:
        return

    final_epoch = max(epochs)
    rows = []
    for group in concept_groups(vocab):
        base = group_df.filter((pl.col("group") == group) & (pl.col("epoch") == -1))
        final = group_df.filter((pl.col("group") == group) & (pl.col("epoch") == final_epoch))
        if len(base) and len(final):
            rows.append(
                (
                    group,
                    float(final["mean_similarity"][0] - base["mean_similarity"][0]),
                    vocab.group_colors.get(group, "#555555"),
                )
            )

    if not rows:
        return

    rows = sorted(rows, key=lambda item: item[1], reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_title(f"Language group drift: epoch {final_epoch} minus baseline")
    ax.set_ylabel("Delta mean cosine similarity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = artifact_root / "language_group_delta.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_language_concept_deltas(artifact_root: Path, concept_rows: list[dict], vocab: LanguageVocab) -> None:
    df = pl.DataFrame(concept_rows)
    epochs = sorted(df["epoch"].unique().to_list())
    if -1 not in epochs or len(epochs) < 2:
        return

    final_epoch = max(epochs)
    wide = (
        df.filter(pl.col("epoch").is_in([-1, final_epoch]))
        .pivot(index=["concept", "group"], on="epoch", values="mean_similarity")
        .rename({"-1": "baseline", str(final_epoch): "final"})
        .with_columns((pl.col("final") - pl.col("baseline")).alias("delta"))
    )

    top_pos = wide.sort("delta", descending=True).head(12)
    top_neg = wide.sort("delta", descending=False).head(12)
    plot_df = pl.concat([top_neg, top_pos]).sort("delta")

    labels = [f"{c[:34]} ({g})" for c, g in zip(plot_df["concept"], plot_df["group"])]
    values = plot_df["delta"].to_numpy()
    colors = [vocab.group_colors.get(g, "#555555") for g in plot_df["group"]]

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Delta mean cosine similarity")
    ax.set_title(f"Top language concept changes: epoch {final_epoch} minus baseline")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = artifact_root / "language_top_concept_deltas.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_language_centered_concept_deltas(artifact_root: Path, concept_rows: list[dict], vocab: LanguageVocab) -> None:
    df = pl.DataFrame(concept_rows)
    epochs = sorted(df["epoch"].unique().to_list())
    if -1 not in epochs or len(epochs) < 2:
        return

    final_epoch = max(epochs)
    wide = (
        df.filter(pl.col("epoch").is_in([-1, final_epoch]))
        .pivot(index=["concept", "group"], on="epoch", values="mean_similarity")
        .rename({"-1": "baseline", str(final_epoch): "final"})
        .with_columns((pl.col("final") - pl.col("baseline")).alias("delta"))
        .with_columns((pl.col("delta") - pl.col("delta").mean()).alias("centered_delta"))
    )

    top_pos = wide.sort("centered_delta", descending=True).head(14)
    top_neg = wide.sort("centered_delta", descending=False).head(14)
    plot_df = pl.concat([top_neg, top_pos]).sort("centered_delta")

    labels = [f"{c[:34]} ({g})" for c, g in zip(plot_df["concept"], plot_df["group"])]
    values = plot_df["centered_delta"].to_numpy()
    colors = [vocab.group_colors.get(g, "#555555") for g in plot_df["group"]]

    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Centered delta")
    ax.set_title(f"Top relative language concept changes: epoch {final_epoch} minus baseline")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = artifact_root / "language_top_centered_concept_deltas.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def final_delta_table(concept_rows: list[dict]) -> tuple[pl.DataFrame, int] | tuple[None, None]:
    df = pl.DataFrame(concept_rows)
    epochs = sorted(df["epoch"].unique().to_list())
    if -1 not in epochs or len(epochs) < 2:
        return None, None

    final_epoch = max(epochs)
    wide = (
        df.filter(pl.col("epoch").is_in([-1, final_epoch]))
        .pivot(index=["concept", "group"], on="epoch", values="mean_similarity")
        .rename({"-1": "baseline", str(final_epoch): "final"})
        .with_columns((pl.col("final") - pl.col("baseline")).alias("delta"))
        .with_columns((pl.col("delta") - pl.col("delta").mean()).alias("centered_delta"))
    )
    return wide, final_epoch


def plot_language_delta_distribution(artifact_root: Path, concept_rows: list[dict]) -> None:
    wide, final_epoch = final_delta_table(concept_rows)
    if wide is None:
        return

    raw = wide["delta"].to_numpy()
    centered = wide["centered_delta"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(raw, bins=45, color="#4c78a8", alpha=0.85)
    axes[0].axvline(0, color="#222222", linewidth=0.9)
    axes[0].axvline(raw.mean(), color="#d95f02", linewidth=1.4, label="mean")
    axes[0].set_title(f"Raw concept deltas: epoch {final_epoch} minus baseline")
    axes[0].set_xlabel("Raw delta")
    axes[0].set_ylabel("Concept count")
    axes[0].legend()

    axes[1].hist(centered, bins=45, color="#59a14f", alpha=0.85)
    axes[1].axvline(0, color="#222222", linewidth=0.9)
    axes[1].set_title("Centered deltas after removing global shift")
    axes[1].set_xlabel("Centered delta")
    axes[1].set_ylabel("Concept count")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = artifact_root / "language_delta_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_language_baseline_final_scatter(artifact_root: Path, concept_rows: list[dict], vocab: LanguageVocab) -> None:
    wide, final_epoch = final_delta_table(concept_rows)
    if wide is None:
        return

    baseline = wide["baseline"].to_numpy()
    final = wide["final"].to_numpy()
    groups = wide["group"].to_list()
    colors = [vocab.group_colors.get(group, "#777777") for group in groups]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(baseline, final, s=14, c=colors, alpha=0.65, linewidths=0)
    low = min(float(baseline.min()), float(final.min()))
    high = max(float(baseline.max()), float(final.max()))
    ax.plot([low, high], [low, high], color="#222222", linewidth=0.9, label="no drift")
    ax.set_title(f"All concepts: baseline vs epoch {final_epoch} similarity")
    ax.set_xlabel("Baseline mean similarity")
    ax.set_ylabel(f"Epoch {final_epoch} mean similarity")
    ax.grid(alpha=0.25)

    handles = []
    for group in concept_groups(vocab):
        if group in groups:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=vocab.group_colors.get(group, "#777777"),
                    label=group,
                    markersize=6,
                )
            )
    ax.legend(handles=handles, fontsize=8, loc="best")

    fig.tight_layout()
    out = artifact_root / "language_baseline_vs_final_scatter.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_language_group_boxplot(artifact_root: Path, concept_rows: list[dict], vocab: LanguageVocab) -> None:
    wide, final_epoch = final_delta_table(concept_rows)
    if wide is None:
        return

    groups = [group for group in concept_groups(vocab) if len(wide.filter(pl.col("group") == group)) > 0]
    values = [wide.filter(pl.col("group") == group)["centered_delta"].to_numpy() for group in groups]

    fig, ax = plt.subplots(figsize=(max(8, 1.05 * len(groups)), 5.2))
    parts = ax.boxplot(values, tick_labels=groups, patch_artist=True, showfliers=False)
    for patch, group in zip(parts["boxes"], groups):
        patch.set_facecolor(vocab.group_colors.get(group, "#777777"))
        patch.set_alpha(0.72)
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_title(f"Centered concept drift by group: epoch {final_epoch} minus baseline")
    ax.set_ylabel("Centered delta")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    out = artifact_root / "language_group_centered_delta_boxplot.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_language_all_concepts_heatmap(artifact_root: Path, concept_rows: list[dict]) -> None:
    df = pl.DataFrame(concept_rows)
    epochs = sorted(df["epoch"].unique().to_list())
    if -1 not in epochs or len(epochs) < 2:
        return

    wide = (
        df.pivot(index=["concept", "group"], on="epoch", values="mean_similarity")
        .rename({"-1": "base"})
    )
    epoch_cols = ["base"] + [str(e) for e in epochs if e != -1]
    if any(col not in wide.columns for col in epoch_cols):
        return

    scores = wide.select(epoch_cols).to_numpy().astype(np.float32)
    delta = scores - scores[:, [0]]
    centered = delta - delta.mean(axis=0, keepdims=True)
    final_centered = centered[:, -1]
    order = np.argsort(final_centered)
    matrix = centered[order]

    fig_height = min(18, max(7, len(order) * 0.018))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    vmax = max(0.01, float(np.percentile(np.abs(matrix), 99)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(epoch_cols)))
    ax.set_xticklabels(epoch_cols)
    ax.set_yticks([])
    ax.set_title("All concepts sorted by final centered drift")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Concepts, low to high final centered drift")
    fig.colorbar(im, ax=ax, label="Centered delta from baseline")
    fig.tight_layout()
    out = artifact_root / "language_all_concepts_centered_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float32)
    return ranks


def plot_language_rank_stability(artifact_root: Path, concept_rows: list[dict]) -> None:
    df = pl.DataFrame(concept_rows)
    epochs = sorted(df["epoch"].unique().to_list())
    if -1 not in epochs or len(epochs) < 2:
        return

    concept_order = sorted(df["concept"].unique().to_list())
    base = (
        df.filter(pl.col("epoch") == -1)
        .select(["concept", "mean_similarity"])
        .sort("concept")
    )
    if base["concept"].to_list() != concept_order:
        return

    base_rank = rank_desc(base["mean_similarity"].to_numpy())
    rows = []
    for epoch in epochs:
        current = (
            df.filter(pl.col("epoch") == epoch)
            .select(["concept", "mean_similarity"])
            .sort("concept")
        )
        current_rank = rank_desc(current["mean_similarity"].to_numpy())
        corr = float(np.corrcoef(base_rank, current_rank)[0, 1])
        rows.append(("base" if epoch == -1 else str(epoch), corr))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", color="#444444")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Language concept rank stability vs baseline")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Rank correlation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = artifact_root / "language_rank_stability.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_language_top_drift_heatmap(artifact_root: Path, concept_rows: list[dict], top_k: int) -> None:
    df = pl.DataFrame(concept_rows)
    epochs = sorted(df["epoch"].unique().to_list())
    if -1 not in epochs or len(epochs) < 2:
        return

    final_epoch = max(epochs)
    wide = (
        df.pivot(index=["concept", "group"], on="epoch", values="mean_similarity")
        .rename({"-1": "base"})
    )
    final_col = str(final_epoch)
    if "base" not in wide.columns or final_col not in wide.columns:
        return

    wide = wide.with_columns((pl.col(final_col) - pl.col("base")).abs().alias("abs_delta"))
    selected = wide.sort("abs_delta", descending=True).head(max(8, top_k))
    labels = [f"{c[:32]} ({g})" for c, g in zip(selected["concept"], selected["group"])]
    epoch_cols = ["base"] + [str(e) for e in epochs if e != -1]
    matrix = selected.select(epoch_cols).to_numpy().astype(np.float32)
    matrix = matrix - matrix[:, [0]]

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.32 * len(labels))))
    vmax = max(0.01, float(np.abs(matrix).max()))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(epoch_cols)))
    ax.set_xticklabels(epoch_cols)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Top drifting language concepts, baseline-centered")
    ax.set_xlabel("Checkpoint")
    fig.colorbar(im, ax=ax, label="Delta from baseline")
    fig.tight_layout()
    out = artifact_root / "language_top_drift_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_language_centered_drift_heatmap(artifact_root: Path, concept_rows: list[dict], top_k: int) -> None:
    df = pl.DataFrame(concept_rows)
    epochs = sorted(df["epoch"].unique().to_list())
    if -1 not in epochs or len(epochs) < 2:
        return

    final_epoch = max(epochs)
    wide = (
        df.pivot(index=["concept", "group"], on="epoch", values="mean_similarity")
        .rename({"-1": "base"})
    )
    epoch_cols = ["base"] + [str(e) for e in epochs if e != -1]
    if any(col not in wide.columns for col in epoch_cols):
        return

    scores = wide.select(epoch_cols).to_numpy().astype(np.float32)
    delta = scores - scores[:, [0]]
    centered = delta - delta.mean(axis=0, keepdims=True)
    final_index = epoch_cols.index(str(final_epoch))
    selected_idx = np.argsort(np.abs(centered[:, final_index]))[::-1][: max(8, top_k)]

    rank_by_row = {int(row): rank for rank, row in enumerate(selected_idx.tolist())}
    selected = (
        wide.with_row_index("_row")
        .filter(pl.col("_row").is_in(selected_idx.tolist()))
        .with_columns(pl.col("_row").replace_strict(rank_by_row).alias("_rank"))
        .sort("_rank")
    )
    labels = [f"{c[:32]} ({g})" for c, g in zip(selected["concept"], selected["group"])]
    matrix = centered[selected["_row"].to_numpy()]

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.32 * len(labels))))
    vmax = max(0.01, float(np.abs(matrix).max()))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(epoch_cols)))
    ax.set_xticklabels(epoch_cols)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Top relative drifting language concepts, global shift removed")
    ax.set_xlabel("Checkpoint")
    fig.colorbar(im, ax=ax, label="Centered delta from baseline")
    fig.tight_layout()
    out = artifact_root / "language_top_centered_drift_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_language_anchor_tsne(
    args: argparse.Namespace,
    artifact_root: Path,
    caches: list[EmbeddingCache],
    text_np: np.ndarray,
    concept_group: np.ndarray,
    vocab: LanguageVocab,
) -> None:
    ordered = ordered_caches(caches)
    if len(ordered) < 2:
        return

    baseline = ordered[0]
    final = ordered[-1]
    base_vision, _ = load_embeddings(baseline, "vision")
    final_vision, _ = load_embeddings(final, "vision")

    rng = np.random.default_rng(args.seed)
    max_points = min(len(base_vision), len(final_vision), 600)
    paired_count = min(len(base_vision), len(final_vision))
    idx = (
        rng.choice(paired_count, size=max_points, replace=False)
        if paired_count > max_points
        else np.arange(paired_count)
    )

    anchor_indices = []
    for group in concept_groups(vocab):
        group_idx = np.where(concept_group == group)[0]
        if len(group_idx) > args.language_top_k:
            anchor_indices.extend(group_idx[: args.language_top_k].tolist())
        else:
            anchor_indices.extend(group_idx.tolist())
    anchor_indices = np.array(anchor_indices, dtype=int)

    arrays = [
        l2_normalize(base_vision[idx]),
        l2_normalize(final_vision[idx]),
        text_np[anchor_indices],
    ]
    all_embeddings = np.concatenate(arrays, axis=0)

    n_components = min(50, all_embeddings.shape[1], max(2, len(all_embeddings) - 1))
    reduced = PCA(n_components=n_components, random_state=args.seed).fit_transform(all_embeddings)
    tsne = TSNE(
        n_components=2,
        random_state=args.seed,
        perplexity=min(args.perplexity, max(1.0, (len(all_embeddings) - 1) / 3), len(all_embeddings) - 1.0),
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(reduced)

    n_base = len(arrays[0])
    n_final = len(arrays[1])
    base_coords = coords[:n_base]
    final_coords = coords[n_base : n_base + n_final]
    text_coords = coords[n_base + n_final :]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(base_coords[:, 0], base_coords[:, 1], s=8, c="#777777", alpha=0.35, label="baseline video")
    ax.scatter(final_coords[:, 0], final_coords[:, 1], s=8, c="#1f77b4", alpha=0.35, label=f"{final.tag} video")

    for group in concept_groups(vocab):
        mask = concept_group[anchor_indices] == group
        ax.scatter(
            text_coords[mask, 0],
            text_coords[mask, 1],
            s=45,
            marker="x",
            c=vocab.group_colors.get(group, "#111111"),
            label=f"{group} text",
        )

    ax.set_title("Video embeddings relative to fixed SigLIP2 text anchors")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = artifact_root / "language_anchor_tsne.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def save_run_manifest(args: argparse.Namespace, artifact_root: Path, caches: list[EmbeddingCache]) -> None:
    manifest = {
        "experiment": args.experiment,
        "experiment_dir": str(Path(args.experiment_dir).expanduser()) if args.experiment_dir else None,
        "checkpoint_dir": args.checkpoint_dir,
        "epochs": args.epochs,
        "all_epochs": args.all_epochs,
        "tsne_epochs": args.tsne_epochs,
        "partition": args.partition,
        "max_observations": args.max_observations,
        "batch_size": args.batch_size,
        "spaces": args.spaces,
        "run_language": args.run_language,
        "language_vocab": args.language_vocab,
        "language_max_concepts": args.language_max_concepts,
        "language_prompt_mode": args.language_prompt_mode,
        "language_group_mode": args.language_group_mode,
        "language_only": args.language_only,
        "caches": [
            {
                "tag": cache.tag,
                "epoch": cache.epoch,
                "embeddings": str(cache.npz_path),
                "metadata": str(cache.metadata_path),
            }
            for cache in caches
        ],
    }
    manifest_name = "run_manifest_language.json" if args.language_only else "run_manifest.json"
    with (artifact_root / manifest_name).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, data_base, exp_dir, artifact_root = get_paths(args)
    if args.all_epochs:
        args.epochs = discover_checkpoint_epochs(get_checkpoint_dir(args, exp_dir))
        print(f"Discovered epochs: {args.epochs}")
    cfg = load_cfg(args, exp_dir, data_base)

    epochs: list[int | None] = [None] + args.epochs
    caches = []
    for epoch in epochs:
        cache = cache_paths(artifact_root, epoch)
        if args.skip_embedding:
            if not cache.npz_path.exists() or not cache.metadata_path.exists():
                raise FileNotFoundError(f"Missing cache for {cache.tag}; rerun without --skip-embedding")
            caches.append(cache)
        else:
            caches.append(cache_embeddings(args, cfg, exp_dir, artifact_root, epoch))

    if not args.language_only:
        write_alignment_and_plots(args, cfg, artifact_root, caches)

    if args.run_language:
        compute_language_metrics(args, artifact_root, caches)

    save_run_manifest(args, artifact_root, caches)
    print(f"Done. Outputs: {artifact_root.resolve()}")


if __name__ == "__main__":
    main()
