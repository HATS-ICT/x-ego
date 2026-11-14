#!/usr/bin/env python3
"""
Union of observer windows to encode, split into priority tiers.

  tier `core`: the 8 historical Team/Enemy occupancy tasks, self_speed,
               teammate_speed, and the matched Self/teammate/enemy suite
  tier `rest`: every other historical task (full benchmark + P5 restrictions)

A window already in `core` is never repeated in `rest`. Each window is encoded
once per encoder and joined back to task rows by `window_id`.

    uv run python -m src.scripts.iclr2027.build_window_manifest --map mirage
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from rich import print as rprint

from .split_common import COHORT, cohort_dir, ensure_tick_norm, sha256_file, tasks_folder, window_id, write_manifest

DERIVED: list[str] = []

CORE_TASKS = [
    *(f"teammate_location_{h}s" for h in (0, 5, 10, 20)),
    *(f"enemy_location_{h}s" for h in (0, 5, 10, 20)),
    "self_speed",
    "teammate_speed",
]
KEYS = ["match_id", "round_num", "pov_steamid", "start_tick_norm", "end_tick_norm", "partition"]


def task_windows(path: Path) -> pl.DataFrame:
    df, derived = ensure_tick_norm(pl.read_csv(path, null_values=[]), path.parents[2].parent, path.parents[2].name)
    if derived:
        DERIVED.append(path.stem)
    return df.select(
        pl.col("match_id").cast(pl.Utf8),
        pl.col("round_num").cast(pl.Int64),
        pl.col("pov_steamid").cast(pl.Int64).cast(pl.Utf8),
        pl.col("start_tick_norm").cast(pl.Float64).round(0).cast(pl.Int64),
        pl.col("end_tick_norm").cast(pl.Float64).round(0).cast(pl.Int64),
        pl.col("partition"),
    )


def with_ids(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique(subset=KEYS[:5]).with_columns(
        pl.struct(KEYS[:5]).map_elements(
            lambda r: window_id(r["match_id"], r["round_num"], r["pov_steamid"], r["start_tick_norm"], r["end_tick_norm"]),
            return_dtype=pl.Utf8,
        ).alias("window_id")
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="mirage")
    args = parser.parse_args()
    data_root = Path(os.environ["DATA_BASE_PATH"])
    labels = data_root / args.map / "labels"
    tasks_dir = labels / tasks_folder()

    matched = pl.read_csv(labels / f"matched_suite_{COHORT}.csv", null_values=[]).select(
        pl.col("match_id").cast(pl.Utf8), pl.col("round_num").cast(pl.Int64),
        pl.col("pov_steamid").cast(pl.Utf8), pl.col("start_tick_norm").cast(pl.Int64),
        pl.col("end_tick_norm").cast(pl.Int64), pl.col("partition"),
    )
    core = with_ids(pl.concat([task_windows(tasks_dir / f"{t}.csv") for t in CORE_TASKS] + [matched]))
    rest_tasks = sorted(p.stem for p in tasks_dir.glob("*.csv") if p.stem not in CORE_TASKS)
    rest = with_ids(pl.concat([task_windows(tasks_dir / f"{t}.csv") for t in rest_tasks]))
    rest = rest.join(core.select("window_id"), on="window_id", how="anti")

    # One window must map to exactly one partition (match-held-out guarantees it).
    for name, df in (("core", core), ("rest", rest)):
        if df["window_id"].n_unique() != df.height:
            raise ValueError(f"{name}: duplicate window_id with conflicting partitions")

    manifest = pl.concat([core.with_columns(pl.lit("core").alias("tier")), rest.with_columns(pl.lit("rest").alias("tier"))])
    out = cohort_dir(data_root, args.map) / "feature_windows.csv"
    manifest.select(["window_id", "tier", *KEYS]).write_csv(out)
    write_manifest(
        cohort_dir(data_root, args.map) / "feature_windows_manifest.json",
        {
            "cohort": COHORT,
            "core_tasks": CORE_TASKS + ["matched_suite"],
            "rest_tasks": rest_tasks,
            "windows": {"core": core.height, "rest": rest.height},
            "tick_norm_derived_from_metadata": sorted(set(DERIVED)),
            "file": out.name,
            "sha256": sha256_file(out),
        },
    )
    rprint(f"[green]OK[/green] {out}: core={core.height:,} rest={rest.height:,}")


if __name__ == "__main__":
    main()
