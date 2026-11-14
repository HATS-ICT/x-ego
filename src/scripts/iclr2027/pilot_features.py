#!/usr/bin/env python3
"""
Pilot gate for the feature cache.

1. Input equivalence: for real rows of historical task CSVs, the clip decoded by
   the historical DownstreamDataset must be bit-identical to the clip decoded by
   the feature extractor for the same window_id. (Encoder weights are loaded the
   same way, so equal inputs => equal h.)
2. Determinism: encoding the same windows twice gives identical h.
3. Encoders differ: adapted checkpoints do not reproduce the frozen features.
4. Throughput: windows/s with all encoders, for shard sizing.

    uv run python -m src.scripts.iclr2027.pilot_features
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from rich import print as rprint

from src.dataset.downstream import DownstreamDataset
from src.utils.config_utils import apply_task_config, load_cfg

from .extract_features import WindowDataset, build_cfg, discover_encoders, load_encoder
from .split_common import cohort_dir, tasks_folder, window_id

TASKS = ["teammate_location_5s", "self_speed", "enemy_location_0s"]


def main() -> None:
    load_dotenv()
    data_root = Path(os.environ["DATA_BASE_PATH"])
    cfg = build_cfg()
    windows = pl.read_csv(cohort_dir(data_root, "mirage") / "feature_windows.csv", schema_overrides={"pov_steamid": pl.Utf8})
    by_id = {r["window_id"]: r for r in windows.to_dicts()}
    wds_all = WindowDataset(cfg, windows)
    pos = {r["window_id"]: i for i, r in enumerate(wds_all.rows)}

    # ---- 1. input equivalence against the historical loader ----
    checked = 0
    for task in TASKS:
        dcfg = load_cfg("configs/train/downstream.yaml")
        dcfg = OmegaConf.merge(dcfg, OmegaConf.create({
            "path": {"data": str(data_root)},
            "task": {"task_id": task},
            "data": {"map": "mirage", "labels_folder": tasks_folder(), "partition": "all"},
        }))
        dcfg = apply_task_config(dcfg, data_root)
        dcfg.data.label_path = str(data_root / "mirage" / "labels" / tasks_folder() / dcfg.data.labels_filename)
        dds = DownstreamDataset(dcfg)
        for i in np.random.default_rng(0).choice(len(dds), size=6, replace=False):
            row = dds.df.row(int(i), named=True)
            wid = window_id(row["match_id"], row["round_num"], str(int(row["pov_steamid"])),
                            int(round(row["start_tick_norm"])), int(round(row["end_tick_norm"])))
            if wid not in by_id:
                raise AssertionError(f"{task} row {i}: window {wid} missing from feature_windows.csv")
            hist = dds[int(i)]["video"]
            mine = wds_all[pos[wid]][0]
            if hist.shape != mine.shape or not torch.equal(hist, mine):
                raise AssertionError(f"{task} row {i}: decoded clip differs (max |d|={(hist.float()-mine.float()).abs().max():.4g})")
            checked += 1
    rprint(f"[green]PASS[/green] input equivalence: {checked} task rows bit-identical to the historical loader")

    # ---- 2-4. determinism, encoder difference, throughput ----
    encoders = discover_encoders(Path(os.environ["OUTPUT_BASE_PATH"]), [1, 2, 3])
    rprint(f"encoders found: {sorted(encoders)}")
    device = torch.device("cuda")
    models = {n: load_encoder(cfg, c).to(device) for n, c in encoders.items()}
    batch = torch.stack([wds_all[i][0] for i in range(32)]).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h1 = {n: m(batch).float() for n, m in models.items()}
        h2 = {n: m(batch).float() for n, m in models.items()}
    det = max((h1[n] - h2[n]).abs().max().item() for n in models)
    rprint(f"{'PASS' if det == 0 else 'WARN'} determinism: max |h1-h2| = {det:.3g}")
    for n in models:
        if n == "frozen":
            continue
        d = (h1[n] - h1["frozen"]).norm(dim=1).mean() / h1["frozen"].norm(dim=1).mean()
        rprint(f"  {n:12s} relative distance from frozen: {d:.3f}")
        if d < 1e-3:
            raise AssertionError(f"{n} features equal frozen features: checkpoint not loaded")

    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(wds_all, range(640)), batch_size=64, num_workers=12)
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for video, _, _ in loader:
            video = video.to(device)
            for m in models.values():
                m(video)
    torch.cuda.synchronize()
    rate = 640 / (time.time() - t0)
    core = windows.filter(pl.col("tier") == "core").height
    rest = windows.filter(pl.col("tier") == "rest").height
    rprint(f"throughput: {rate:.1f} windows/s with {len(models)} encoders "
           f"-> core {core:,} = {core / rate / 3600:.2f} h, rest {rest:,} = {rest / rate / 3600:.2f} h on this GPU")


if __name__ == "__main__":
    main()
