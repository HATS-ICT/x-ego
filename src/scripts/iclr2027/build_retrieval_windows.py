#!/usr/bin/env python3
"""
Tier-2 diagnostic windows: complete TEST-partition team rows, so every clip's
true synchronized teammates are in the pool. Sampled uniformly over team rows
with a fixed seed. Written to its own file; the core/rest manifest is untouched.

    uv run python -m src.scripts.iclr2027.build_retrieval_windows --n-rows 1500
"""
import argparse
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from .split_common import TICK_RATE, cohort_dir, contrastive_filename, sha256_file, window_id, write_manifest


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="mirage")
    ap.add_argument("--n-rows", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()
    data_root = Path(os.environ["DATA_BASE_PATH"])
    df = pl.read_csv(data_root / args.map / "labels" / contrastive_filename(), null_values=[])
    rows = df.filter(pl.col("partition") == "test").sample(n=args.n_rows, seed=args.seed)
    out = []
    for r in rows.iter_rows(named=True):
        s, e = int(round(r["start_seconds"] * TICK_RATE)), int(round(r["end_seconds"] * TICK_RATE))
        for k in range(int(r["num_alive_teammates"])):
            pid = str(r[f"teammate_{k}_id"])
            out.append({"window_id": window_id(r["match_id"], r["round_num"], pid, s, e), "tier": "retr",
                        "match_id": r["match_id"], "round_num": int(r["round_num"]), "pov_steamid": pid,
                        "start_tick_norm": s, "end_tick_norm": e, "partition": "test",
                        "group_id": int(r["idx"]), "pov_side": str(r["pov_team_side"])})
    w = pl.DataFrame(out)
    path = cohort_dir(data_root, args.map) / "retrieval_windows.csv"
    w.write_csv(path)
    write_manifest(cohort_dir(data_root, args.map) / "retrieval_windows_manifest.json",
                   {"team_rows": rows.height, "clips": w.height, "matches": w["match_id"].n_unique(),
                    "seed": args.seed, "sha256": sha256_file(path)})
    print(f"{path}: {rows.height} team rows, {w.height} clips, {w['match_id'].n_unique()} matches")


if __name__ == "__main__":
    main()
