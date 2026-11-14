#!/usr/bin/env python3
"""
P2-B -- matched-perspective task suite on shared observer windows.

For each sampled observer window this writes, on the SAME window:
  * region of Self, one specified teammate and one specified enemy, at the
    input endpoint (0 s ahead) and 5 / 10 s after it          -> 9 region tasks
  * endpoint speed of Self and of the specified teammate      -> 2 speed tasks

Frozen rules:
  * windows are observer clips of the match-held-out cohort, sampled uniformly
    with a fixed seed (so partitions keep their clip proportions);
  * the specified teammate / enemy is the LOWEST steamid among players of that
    side alive at the input endpoint (teammate excludes the observer). Only
    information available at the endpoint is used -- never future survival,
    proximity or visibility;
  * a target is invalid at horizon h if the round has ended, the target is
    dead / has no trajectory at that tick, or its place is outside the map
    vocabulary. Every exclusion is counted by task and horizon;
  * speed = |position(t_end) - position(t_end - 0.5 s)| / 0.5 s in RAW game
    units per second, for Self and teammate alike. (The historical speed labels
    were rescaled after creation; R^2 is scale-invariant, and MAE is reported in
    these documented units.)
  * the historical midpoint tasks are untouched and kept separate.

Usage:
    uv run python -m src.scripts.iclr2027.build_matched_labels --map mirage --n-windows 20000
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from rich import print as rprint

from src.scripts.task_creator.task_definitions import get_place_to_idx_for_map

from .split_common import (
    COHORT,
    TICK_RATE,
    cohort_dir,
    contrastive_filename,
    git_commit,
    sha256_file,
    window_id,
    write_manifest,
)

HORIZONS_SEC = (0, 5, 10)
SPEED_LOOKBACK_SEC = 0.5
PERSPECTIVES = ("self", "teammate", "enemy")
SAMPLE_SEED = 20260923


class RoundTrajectories:
    """All players' trajectories for one round, indexed by absolute tick."""

    def __init__(self, traj_root: Path, match_id: str, round_num: int):
        self.players: dict[str, pd.DataFrame] = {}
        match_dir = traj_root / match_id
        if not match_dir.exists():
            return
        for player_dir in match_dir.iterdir():
            f = player_dir / f"round_{round_num}.csv"
            if f.exists():
                df = pd.read_csv(f, keep_default_na=False, usecols=["tick", "side", "X", "Y", "Z", "place", "health"])
                if not df.empty:
                    self.players[player_dir.name] = df.set_index("tick", drop=False).sort_index()

    def at(self, steamid: str, tick: int):
        df = self.players.get(str(steamid))
        if df is None or tick not in df.index:
            return None
        row = df.loc[tick]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def alive(self, steamid: str, tick: int) -> bool:
        row = self.at(steamid, tick)
        return row is not None and float(row["health"]) > 0

    def side_of(self, steamid: str, tick: int):
        row = self.at(steamid, tick)
        return None if row is None else str(row["side"]).lower()


def sample_windows(contrastive: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    rows = []
    for r in contrastive.iter_rows(named=True):
        for slot in range(int(r["num_alive_teammates"])):
            rows.append(
                {
                    "partition": r["partition"],
                    "match_id": r["match_id"],
                    "round_num": int(r["round_num"]),
                    "pov_steamid": str(r[f"teammate_{slot}_id"]),
                    "pov_side": str(r["pov_team_side"]).lower(),
                    "start_tick_norm": int(round(r["start_seconds"] * TICK_RATE)),
                    "end_tick_norm": int(round(r["end_seconds"] * TICK_RATE)),
                }
            )
    clips = pl.DataFrame(rows)
    n = min(n, clips.height)
    return clips.sample(n=n, seed=seed, shuffle=True).sort(["match_id", "round_num", "start_tick_norm", "pov_steamid"])


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build the matched Self/teammate/enemy suite (P2-B)")
    parser.add_argument("--map", default="mirage")
    parser.add_argument("--n-windows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args()

    repo_root = Path(os.getenv("SRC_BASE_PATH", ".")).resolve()
    data_root = Path(os.environ["DATA_BASE_PATH"])
    map_dir = data_root / args.map
    place_to_idx = get_place_to_idx_for_map(f"de_{args.map}")

    contrastive = pl.read_csv(map_dir / "labels" / contrastive_filename(), null_values=[])
    windows = sample_windows(contrastive, args.n_windows, args.seed)
    rprint(f"[blue]->[/blue] sampled {windows.height:,} observer windows; vocabulary {len(place_to_idx)} places")

    meta_cache: dict[str, dict] = {}
    exclusions: dict[str, Counter] = defaultdict(Counter)
    out = []

    for (match_id, round_num), block in windows.group_by(["match_id", "round_num"], maintain_order=True):
        if match_id not in meta_cache:
            meta_cache[match_id] = json.loads((map_dir / "metadata" / f"{match_id}.json").read_text())
        rinfo = next((r for r in meta_cache[match_id]["rounds"] if r["round_number"] == round_num), None)
        traj = RoundTrajectories(map_dir / "trajectory", match_id, round_num)

        for w in block.iter_rows(named=True):
            rec = dict(w)
            rec["window_id"] = window_id(match_id, round_num, w["pov_steamid"], w["start_tick_norm"], w["end_tick_norm"])
            if rinfo is None or not traj.players:
                for p in PERSPECTIVES:
                    for h in HORIZONS_SEC:
                        rec[f"{p}_region_{h}s"] = None
                        exclusions[f"{p}_region_{h}s"]["missing_round_or_trajectory"] += 1
                rec["self_speed_0s"] = rec["teammate_speed_0s"] = None
                exclusions["self_speed_0s"]["missing_round_or_trajectory"] += 1
                exclusions["teammate_speed_0s"]["missing_round_or_trajectory"] += 1
                rec["teammate_steamid"] = rec["enemy_steamid"] = None
                out.append(rec)
                continue

            freeze_end = int(rinfo["freeze_end_tick"])
            round_end = int(rinfo["end_tick"])
            t_end = freeze_end + w["end_tick_norm"]
            rec["start_tick"] = freeze_end + w["start_tick_norm"]
            rec["end_tick"] = t_end

            # Roster-slot rule: lowest steamid alive at the input endpoint.
            pov_side = traj.side_of(w["pov_steamid"], t_end) or w["pov_side"]
            same, other = [], []
            for sid in traj.players:
                if sid == w["pov_steamid"] or not traj.alive(sid, t_end):
                    continue
                side = traj.side_of(sid, t_end)
                (same if side == pov_side else other).append(sid)
            targets = {
                "self": w["pov_steamid"],
                "teammate": min(same, key=int) if same else None,
                "enemy": min(other, key=int) if other else None,
            }
            rec["teammate_steamid"] = targets["teammate"]
            rec["enemy_steamid"] = targets["enemy"]

            for p in PERSPECTIVES:
                for h in HORIZONS_SEC:
                    task = f"{p}_region_{h}s"
                    tick = t_end + h * TICK_RATE
                    reason = None
                    if targets[p] is None:
                        reason = "no_eligible_target_at_endpoint"
                    elif tick > round_end:
                        reason = "round_ended"
                    elif not traj.alive(targets[p], tick):
                        reason = "target_dead_or_missing"
                    else:
                        place = str(traj.at(targets[p], tick)["place"])
                        if place not in place_to_idx:
                            reason = "region_outside_vocabulary"
                    if reason:
                        rec[task] = None
                        exclusions[task][reason] += 1
                    else:
                        rec[task] = place_to_idx[place]
                        exclusions[task]["valid"] += 1

            back = int(SPEED_LOOKBACK_SEC * TICK_RATE)
            for p in ("self", "teammate"):
                task = f"{p}_speed_0s"
                sid = targets[p]
                a = traj.at(sid, t_end - back) if sid else None
                b = traj.at(sid, t_end) if sid else None
                if sid is None:
                    rec[task] = None
                    exclusions[task]["no_eligible_target_at_endpoint"] += 1
                elif a is None or b is None or float(b["health"]) <= 0:
                    rec[task] = None
                    exclusions[task]["target_dead_or_missing"] += 1
                else:
                    d = np.sqrt(sum((float(b[c]) - float(a[c])) ** 2 for c in ("X", "Y", "Z")))
                    rec[task] = d / SPEED_LOOKBACK_SEC
                    exclusions[task]["valid"] += 1
            out.append(rec)

    df = pl.DataFrame(out, infer_schema_length=None)

    # Per-horizon intersection masks: all three perspectives valid on the same window.
    for h in HORIZONS_SEC:
        df = df.with_columns(
            pl.all_horizontal([pl.col(f"{p}_region_{h}s").is_not_null() for p in PERSPECTIVES]).alias(f"matched_valid_{h}s")
        )
    df = df.with_columns(
        (pl.col("self_speed_0s").is_not_null() & pl.col("teammate_speed_0s").is_not_null()).alias("matched_valid_speed")
    )

    out_path = map_dir / "labels" / f"matched_suite_{COHORT}.csv"
    df.write_csv(out_path)

    coverage = {}
    for flag in [f"matched_valid_{h}s" for h in HORIZONS_SEC] + ["matched_valid_speed"]:
        coverage[flag] = {
            part: {
                "valid": int(df.filter((pl.col("partition") == part) & pl.col(flag)).height),
                "candidates": int(df.filter(pl.col("partition") == part).height),
                "matches": int(df.filter((pl.col("partition") == part) & pl.col(flag))["match_id"].n_unique()),
            }
            for part in ("train", "val", "test")
        }
    write_manifest(
        cohort_dir(data_root, args.map) / f"matched_suite_manifest.json",
        {
            "cohort": COHORT,
            "map": args.map,
            "n_windows": df.height,
            "sample_seed": args.seed,
            "horizons_sec": list(HORIZONS_SEC),
            "roster_rule": "lowest steamid among players of the side alive at the input endpoint (teammate excludes observer)",
            "speed_definition": f"|pos(t_end) - pos(t_end - {SPEED_LOOKBACK_SEC}s)| / {SPEED_LOOKBACK_SEC}s, raw game units/s",
            "vocabulary_size": len(place_to_idx),
            "exclusions": {k: dict(v) for k, v in sorted(exclusions.items())},
            "coverage": coverage,
            "file": out_path.name,
            "sha256": sha256_file(out_path),
            "source_sha256": sha256_file(map_dir / "labels" / contrastive_filename()),
            "git_commit": git_commit(repo_root),
        },
    )
    rprint(f"[green]OK[/green] {out_path} ({df.height:,} windows)")
    for flag, parts in coverage.items():
        rprint(f"  {flag:22s} " + "  ".join(f"{p}={v['valid']}/{v['candidates']} ({v['matches']}m)" for p, v in parts.items()))


if __name__ == "__main__":
    main()
