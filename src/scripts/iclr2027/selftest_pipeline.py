#!/usr/bin/env python3
"""End-to-end selftest of probes -> bootstrap -> tables on random 16-d features (tag _selftest)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

load_dotenv()
OUT = Path(os.environ["OUTPUT_BASE_PATH"])
FEAT = OUT / "iclr2027_features/matchheldout_v1/_selftest"
FEAT.mkdir(parents=True, exist_ok=True)
w = pl.read_csv(Path(os.environ["DATA_BASE_PATH"]) / "mirage/splits/matchheldout_v1/feature_windows.csv")
encs = ["frozen"] + [f"{a}_s{s}" for a in ("cecl", "single", "shuffle") for s in (1, 2, 3)]
rng = np.random.default_rng(0)
for tier in ("core", "rest"):
    ids = w.filter(pl.col("tier") == tier)["window_id"].to_numpy().astype("U20")
    np.savez(FEAT / f"{tier}_shard000of001.npz", window_id=ids, encoders=np.array(encs), checkpoints=np.array(["x"] * len(encs)),
             offset_fallback=np.zeros(len(ids), np.int8), **{f"h__{e}": rng.standard_normal((len(ids), 16)).astype(np.float16) for e in encs})
tasks = ["teammate_location_0s", "teammate_location_5s", "teammate_location_10s", "teammate_location_20s",
         "self_speed", "self_region_0s", "self_region_5s", "self_region_10s"]
procs = [subprocess.Popen([sys.executable, "-m", "src.scripts.iclr2027.fit_probes", "--encoder", e, "--tier", "core",
                           "--feat-tag", "_selftest", "--out-tag", "_selftest", "--threads", "1", "--tasks", *tasks])
         for e in encs]
codes = [p.wait() for p in procs]
assert all(c == 0 for c in codes), codes
for cmd in (["src.scripts.iclr2027.analyze_probes", "--out-tag", "_selftest", "--B", "100", "--B-agg", "20", "--workers", "8"],
            ["src.scripts.iclr2027.make_tables", "--out-tag", "_selftest"]):
    subprocess.run([sys.executable, "-m", *cmd], check=True)
an = OUT / "iclr2027_probes/matchheldout_v1/_selftest/_analysis"
agg = pl.read_csv(an / "aggregates.csv")
print(agg.filter(pl.col("kind") == "contrast").select("group", "probe", "metric", "name", "value", "ci_lo", "ci_hi"))
print((an / "summary.md").read_text()[:3000])
print("SELFTEST PASS")
