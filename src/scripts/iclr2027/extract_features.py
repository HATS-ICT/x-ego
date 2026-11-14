#!/usr/bin/env python3
"""
P2 feature cache: decode each observer window ONCE, encode it with every encoder.

Stores the encoder output h (768-d, frame-averaged) -- the exact input of the
paper's linear probes -- NOT the contrastive projector output g(h).

Encoders:  F (unadapted SigLIP2) + every available `last.ckpt` of the P1 runs.
Inputs:    identical to the historical downstream loader (start_tick_norm / 64 +
           per-round offset, 5 s at 4 fps, full UI mask, NO random tube mask,
           NO time jitter), so features are deterministic.

Shards are idempotent and atomic: an existing complete shard is skipped, so the
same shard can be submitted to several partitions and the first run wins.

    uv run python -m src.scripts.iclr2027.extract_features --tier core --shard 0 --num-shards 12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from rich import print as rprint
from torch.utils.data import DataLoader, Dataset

from src.dataset.dataset_utils import construct_video_path, init_video_processor, load_video_clip, transform_video
from src.models.modules.video_encoder import VideoEncoder
from src.utils.config_utils import load_cfg

from .split_common import COHORT, TICK_RATE, cohort_dir

CONFIG = "configs/train/contrastive_iclr2027.yaml"
RUN_GLOB = "mirage_siglip2_mean_{arm}_s{seed}-*/checkpoint/last.ckpt"
ARMS = ("cecl", "single", "shuffle")


def build_cfg():
    cfg = load_cfg(CONFIG)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"path": {
        "data": os.environ["DATA_BASE_PATH"], "output": os.environ["OUTPUT_BASE_PATH"],
        "src": os.environ.get("SRC_BASE_PATH", "."),
    }}))
    # Probe-time inputs: deterministic, no augmentation.
    cfg.data.random_mask.enable = False
    cfg.data.time_jitter_max_seconds = 0
    return cfg


def discover_encoders(output_root: Path, seeds: list[int]) -> dict[str, Path | None]:
    encoders: dict[str, Path | None] = {"frozen": None}
    for arm in ARMS:
        for seed in seeds:
            hits = sorted(glob.glob(str(output_root / RUN_GLOB.format(arm=arm, seed=seed))))
            if len(hits) > 1:
                raise ValueError(f"ambiguous checkpoints for {arm}_s{seed}: {hits}")
            if hits:
                encoders[f"{arm}_s{seed}"] = Path(hits[0])
    return encoders


def load_encoder(cfg, ckpt: Path | None) -> VideoEncoder:
    enc = VideoEncoder(OmegaConf.create(OmegaConf.to_container(cfg.model.encoder)))
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        prefix = "video_encoder."
        sub = {k[len(prefix):].replace("._orig_mod.", "."): v for k, v in sd.items() if k.startswith(prefix)}
        enc.load_state_dict(sub, strict=True)
    if enc.video_encoder.temporal is not None:
        raise AssertionError("temporal block present; P1 protocol requires frame averaging")
    active = [m for m in enc.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
    if active:
        raise AssertionError(f"{len(active)} active dropout layers; cached eval features would differ from probe-time features")
    return enc.eval()


class WindowDataset(Dataset):
    def __init__(self, cfg, windows: pl.DataFrame):
        self.cfg = cfg
        self.rows = windows.to_dicts()
        self.processor, self.processor_type = init_video_processor(cfg)
        data_root = Path(cfg.path.data) / cfg.data.map
        self.offsets = json.loads((data_root / "time_offset.json").read_text())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # Mirrors src/dataset/downstream.py: a missing offset falls back to 0 s.
        # Unlike the historical loader, the fallback is flagged, not silent.
        off = self.offsets.get(str(r["match_id"]), {}).get(str(r["pov_steamid"]), {}).get(f"round_{r['round_num']}")
        offset_sec = float(off["offset_sec"]) if off else 0.0
        start = r["start_tick_norm"] / TICK_RATE + offset_sec
        end = r["end_tick_norm"] / TICK_RATE + offset_sec
        path = construct_video_path(self.cfg, r["match_id"], r["pov_steamid"], int(r["round_num"]))
        clip = load_video_clip(self.cfg, path, start, end)["video"]
        return transform_video(self.processor, self.processor_type, clip), i, int(off is None)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="mirage")
    parser.add_argument("--tier", choices=["core", "rest", "retr"], required=True)
    parser.add_argument("--windows-file", default="feature_windows.csv",
                        help="window manifest inside the cohort split dir (retr uses retrieval_windows.csv)")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--encoders", nargs="*", default=None, help="subset of encoder names (default: all found)")
    parser.add_argument("--require", type=int, default=None,
                        help="fail unless exactly this many encoders are found (guards against a run that has not finished)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="pilot: only the first N windows of the shard")
    parser.add_argument("--out-tag", default="p1")
    args = parser.parse_args()

    cfg = build_cfg()
    data_root = Path(os.environ["DATA_BASE_PATH"])
    out_dir = Path(os.environ["OUTPUT_BASE_PATH"]) / "iclr2027_features" / COHORT / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_pilot{args.limit}" if args.limit else ""
    out_path = out_dir / f"{args.tier}_shard{args.shard:03d}of{args.num_shards:03d}{suffix}.npz"

    encoders = discover_encoders(Path(os.environ["OUTPUT_BASE_PATH"]), args.seeds)
    if args.encoders:
        # Any run name works: "cecl_s1" (P1) or "orig_cecl_s1" (P1b) -> mirage_siglip2_mean_<name>-*
        for name in args.encoders:
            if name in encoders or name == "frozen":
                continue
            hits = sorted(glob.glob(str(Path(os.environ["OUTPUT_BASE_PATH"]) / f"mirage_siglip2_mean_{name}-*/checkpoint/last.ckpt")))
            if len(hits) > 1:
                raise ValueError(f"ambiguous checkpoints for {name}: {hits}")
            if hits:
                encoders[name] = Path(hits[0])
        missing = set(args.encoders) - set(encoders)
        if missing:
            raise ValueError(f"requested encoders not found: {sorted(missing)}")
        encoders = {k: encoders[k] for k in args.encoders}
    if args.require is not None and len(encoders) != args.require:
        raise SystemExit(f"expected {args.require} encoders, found {len(encoders)}: {sorted(encoders)}")

    if out_path.exists():
        with np.load(out_path, allow_pickle=False) as z:
            have = set(z["encoders"].tolist())
        if have >= set(encoders):
            rprint(f"[green]SKIP[/green] {out_path.name} already complete ({len(have)} encoders)")
            return

    windows = pl.read_csv(cohort_dir(data_root, args.map) / args.windows_file, schema_overrides={"pov_steamid": pl.Utf8})
    windows = windows.filter(pl.col("tier") == args.tier).sort("window_id")
    windows = windows.with_row_index("__i").filter(pl.col("__i") % args.num_shards == args.shard).drop("__i")
    if args.limit:
        windows = windows.head(args.limit)
    rprint(f"[blue]->[/blue] {args.tier} shard {args.shard}/{args.num_shards}: {windows.height:,} windows, "
           f"{len(encoders)} encoders: {sorted(encoders)}")

    device = torch.device("cuda")
    models = {name: load_encoder(cfg, ck).to(device) for name, ck in encoders.items()}
    feats = {name: np.zeros((windows.height, 768), dtype=np.float16) for name in models}
    offset_fallback = np.zeros(windows.height, dtype=np.int8)

    # Each decoded clip is ~12 MB (20 x 3 x 224 x 224 float32); workers x prefetch x batch
    # clips are buffered at once, so both are capped to keep the job well inside 64 GB.
    num_workers = min(args.num_workers, 10)
    loader = DataLoader(WindowDataset(cfg, windows), batch_size=args.batch_size, num_workers=num_workers,
                        pin_memory=True, persistent_workers=False, prefetch_factor=2 if num_workers else None)
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b, (video, idx, fallback) in enumerate(loader):
            video = video.to(device, non_blocking=True)
            idx = idx.numpy()
            offset_fallback[idx] = fallback.numpy()
            for name, model in models.items():
                feats[name][idx] = model(video).float().cpu().numpy().astype(np.float16)
            if b % 10 == 0:
                done = min((b + 1) * args.batch_size, windows.height)
                rate = done / max(time.time() - t0, 1e-6)
                rprint(f"  {done:6,d}/{windows.height:,} windows  {rate:5.1f} win/s")

    finite = {n: bool(np.isfinite(f).all()) for n, f in feats.items()}
    if not all(finite.values()):
        raise AssertionError(f"non-finite features: {finite}")

    tmp = out_path.with_name(out_path.name + ".tmp.npz")
    np.savez(
        tmp,
        window_id=windows["window_id"].to_numpy().astype("U20"),
        encoders=np.array(sorted(models), dtype="U32"),
        checkpoints=np.array([str(encoders[n]) for n in sorted(models)], dtype="U512"),
        offset_fallback=offset_fallback,
        **{f"h__{n}": feats[n] for n in sorted(models)},
    )
    os.replace(tmp, out_path)
    rprint(f"[green]OK[/green] {out_path} in {time.time() - t0:.0f}s; offset fallbacks: {int(offset_fallback.sum())}")


if __name__ == "__main__":
    main()
