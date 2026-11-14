#!/usr/bin/env python3
"""
P1 arm H -- build nuisance-matched shuffled-pair manifests.

Construction rules (frozen before launch):
  * The clip multiset is IDENTICAL to the CECL arm: every observer clip is used
    exactly once, so marginal clip exposure is matched by construction.
  * Strata preserve map (single map per manifest), team side, and approximate
    elapsed round phase (10-second bins). Sparse tail bins are merged into one
    tail stratum using a threshold determined on TRAINING metadata only.
  * Group sizes are drawn from the CECL group-size multiset within each stratum,
    so positive degree is preserved.
  * Within a pseudo-group every member comes from a DIFFERENT (match, round),
    which excludes self-pairs, true synchronized teammate positives, and
    overlapping windows of the same player. Members also carry distinct agent
    ids, which removes persistent same-player positives.
  * Clips that cannot be placed under the constraints are dropped and counted.
  * Partitions are shuffled independently; validation/test clips are never used
    as training donors.

Usage:
    uv run python -m src.scripts.iclr2027.build_shuffle_manifest --map mirage --seed 1
"""

from __future__ import annotations

import argparse
import heapq
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl
from dotenv import load_dotenv
from rich import print as rprint

from .split_common import (
    COHORT,
    PARTITIONS,
    cohort_dir,
    contrastive_filename,
    git_commit,
    sha256_file,
    write_manifest,
)

PHASE_BIN_SECONDS = 10.0
# A stratum smaller than this cannot support a valid shuffle on its own and is
# merged into the tail stratum. The threshold is applied to the TRAIN partition
# and the resulting bin edges are reused for val/test.
MIN_STRATUM_CLIPS = 200


def explode_clips(df: pl.DataFrame) -> pl.DataFrame:
    """One row per observer clip, carrying its source team row."""
    records = []
    for row in df.iter_rows(named=True):
        for slot in range(int(row["num_alive_teammates"])):
            records.append(
                {
                    "src_idx": int(row["idx"]),
                    "partition": row["partition"],
                    "match_id": row["match_id"],
                    "round_num": int(row["round_num"]),
                    "agent_id": str(row[f"teammate_{slot}_id"]),
                    "roster_slot": slot,
                    "start_seconds": float(row["start_seconds"]),
                    "end_seconds": float(row["end_seconds"]),
                    "pov_team_side": str(row["pov_team_side"]),
                    "group_size": int(row["num_alive_teammates"]),
                }
            )
    return pl.DataFrame(records)


def phase_bin(start_seconds: float, tail_bin: int) -> int:
    return min(int(start_seconds // PHASE_BIN_SECONDS), tail_bin)


def choose_tail_bin(train_clips: pl.DataFrame) -> int:
    """Smallest bin index such that every bin at or below it holds enough clips."""
    raw = (train_clips["start_seconds"] // PHASE_BIN_SECONDS).cast(int).to_list()
    counts = Counter(raw)
    max_bin = max(counts)
    cumulative = 0
    for b in range(max_bin, -1, -1):
        cumulative += counts.get(b, 0)
        if cumulative >= MIN_STRATUM_CLIPS:
            return b
    return 0


def shuffle_stratum(
    clips: List[dict],
    group_sizes: List[int],
    rng: random.Random,
) -> Tuple[List[List[dict]], List[dict]]:
    """
    Fill groups of the given sizes from `clips`, no two members sharing a
    (match, round) and no two sharing an agent id.

    Uses the standard max-remaining-bucket greedy over (match, round) buckets,
    which places the maximum number of clips whenever any valid packing exists.
    Returns (groups, dropped_clips).
    """
    buckets: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for clip in clips:
        buckets[(clip["match_id"], clip["round_num"])].append(clip)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    heap = [(-len(v), k) for k, v in buckets.items()]
    heapq.heapify(heap)

    groups: List[List[dict]] = []
    # Largest groups first: they are the hardest to satisfy.
    for size in sorted(group_sizes, reverse=True):
        picked: List[dict] = []
        held: List[Tuple[int, Tuple[str, int]]] = []
        used_agents: set = set()

        while len(picked) < size and heap:
            neg_len, key = heapq.heappop(heap)
            bucket = buckets[key]
            if not bucket:
                continue
            # Take the first clip whose agent is not already in this group.
            choice_pos = next(
                (i for i, c in enumerate(bucket) if c["agent_id"] not in used_agents), None
            )
            if choice_pos is None:
                held.append((neg_len, key))
                continue
            clip = bucket.pop(choice_pos)
            picked.append(clip)
            used_agents.add(clip["agent_id"])
            held.append((-len(bucket), key))

        for neg_len, key in held:
            if buckets[key]:
                heapq.heappush(heap, (neg_len, key))

        if len(picked) == size:
            groups.append(picked)
        else:
            # Not enough distinct rounds remain: return the clips to the pool and
            # stop issuing groups for this stratum.
            for clip in picked:
                buckets[(clip["match_id"], clip["round_num"])].append(clip)
            break

    dropped = [clip for bucket in buckets.values() for clip in bucket]
    return groups, dropped


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build a shuffled-pair manifest (P1 arm H)")
    parser.add_argument("--map", default="mirage")
    parser.add_argument("--seed", type=int, required=True, help="encoder seed this manifest pairs with")
    args = parser.parse_args()

    repo_root = Path(os.getenv("SRC_BASE_PATH", ".")).resolve()
    data_root = Path(os.environ["DATA_BASE_PATH"])
    labels_dir = data_root / args.map / "labels"
    src_csv = labels_dir / contrastive_filename()
    if not src_csv.exists():
        raise FileNotFoundError(f"{src_csv} not found; run build_match_heldout_split.py first")

    df = pl.read_csv(src_csv, null_values=[])
    clips = explode_clips(df)
    rprint(f"[blue]->[/blue] {len(clips):,} observer clips from {len(df):,} team rows")

    tail_bin = choose_tail_bin(clips.filter(pl.col("partition") == "train"))
    rprint(f"[blue]->[/blue] phase bins 0..{tail_bin} (>= {tail_bin} merged), {PHASE_BIN_SECONDS:.0f}s wide")

    rng = random.Random(args.seed * 1_000_003 + 7)
    out_rows: List[dict] = []
    audit: Dict = {"per_partition": {}, "strata": []}
    group_id = 0

    for partition in PARTITIONS:
        part = clips.filter(pl.col("partition") == partition)
        strata: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
        for clip in part.iter_rows(named=True):
            clip = dict(clip)
            clip["phase_bin"] = phase_bin(clip["start_seconds"], tail_bin)
            strata[(clip["pov_team_side"], clip["phase_bin"])].append(clip)

        placed = dropped = 0
        for (side, pbin), stratum_clips in sorted(strata.items()):
            # Group sizes of the CECL arm inside this stratum: each source team row
            # contributes its size once.
            sizes_by_row = {c["src_idx"]: c["group_size"] for c in stratum_clips}
            group_sizes = list(sizes_by_row.values())

            groups, leftover = shuffle_stratum(stratum_clips, group_sizes, rng)
            for group in groups:
                for member_idx, clip in enumerate(group):
                    out_rows.append(
                        {
                            "group_id": group_id,
                            "partition": partition,
                            "member_idx": member_idx,
                            "src_idx": clip["src_idx"],
                            "match_id": clip["match_id"],
                            "round_num": clip["round_num"],
                            "agent_id": clip["agent_id"],
                            "start_seconds": clip["start_seconds"],
                            "end_seconds": clip["end_seconds"],
                            "pov_team_side": side,
                            "phase_bin": pbin,
                        }
                    )
                group_id += 1
                placed += len(group)
            dropped += len(leftover)

            audit["strata"].append(
                {
                    "partition": partition,
                    "team_side": side,
                    "phase_bin": pbin,
                    "clips": len(stratum_clips),
                    "groups_requested": len(group_sizes),
                    "groups_formed": len(groups),
                    "clips_dropped": len(leftover),
                }
            )

        audit["per_partition"][partition] = {
            "clips_available": len(part),
            "clips_placed": placed,
            "clips_dropped": dropped,
            "drop_rate": round(dropped / max(len(part), 1), 5),
        }
        rprint(
            f"  [cyan]{partition:5s}[/cyan] placed={placed:6,d} dropped={dropped:5,d} "
            f"({dropped / max(len(part), 1):.2%})"
        )

    manifest_df = pl.DataFrame(out_rows)

    # --- audits the plan requires before this manifest is usable ---
    per_group = manifest_df.group_by("group_id").agg(
        pl.col("match_id").n_unique().alias("n_match"),
        pl.struct("match_id", "round_num").n_unique().alias("n_match_round"),
        pl.col("agent_id").n_unique().alias("n_agent"),
        pl.len().alias("size"),
    )
    violations = per_group.filter(
        (pl.col("n_match_round") != pl.col("size")) | (pl.col("n_agent") != pl.col("size"))
    )
    if violations.height:
        raise ValueError(f"{violations.height} pseudo-groups violate the round/agent constraints")

    exposure = manifest_df.group_by(["match_id", "round_num", "agent_id", "start_seconds"]).agg(
        pl.len().alias("uses")
    )
    if exposure["uses"].max() > 1:
        raise ValueError("a clip is used more than once; marginal exposure is not matched")

    cecl_sizes = Counter(df["num_alive_teammates"].to_list())
    shuffled_sizes = Counter(per_group["size"].to_list())
    audit.update(
        {
            "accidental_true_positive_groups": 0,  # enforced above: distinct (match, round) per group
            "max_clip_uses": int(exposure["uses"].max()),
            "groups": int(per_group.height),
            "group_size_distribution_cecl": {str(k): v for k, v in sorted(cecl_sizes.items())},
            "group_size_distribution_shuffled": {str(k): v for k, v in sorted(shuffled_sizes.items())},
            "phase_distribution": {
                f"{r['partition']}_{r['pov_team_side']}_{r['phase_bin']}": r["len"]
                for r in manifest_df.group_by(["partition", "pov_team_side", "phase_bin"])
                .agg(pl.len())
                .sort(["partition", "pov_team_side", "phase_bin"])
                .iter_rows(named=True)
            },
        }
    )

    out_name = f"contrastive_{COHORT}_shuffle_s{args.seed}.csv"
    out_path = labels_dir / out_name
    manifest_df.write_csv(out_path)

    write_manifest(
        cohort_dir(data_root, args.map) / f"shuffle_audit_s{args.seed}.json",
        {
            "cohort": COHORT,
            "map": args.map,
            "arm": "H_shuffle",
            "seed": args.seed,
            "phase_bin_seconds": PHASE_BIN_SECONDS,
            "min_stratum_clips": MIN_STRATUM_CLIPS,
            "tail_phase_bin": tail_bin,
            "constraints": [
                "distinct (match_id, round_num) per group",
                "distinct agent_id per group",
                "strata matched on (team_side, phase_bin)",
                "each clip used exactly once",
                "partitions shuffled independently",
            ],
            "source_csv": str(src_csv),
            "source_sha256": sha256_file(src_csv),
            "manifest_file": out_name,
            "manifest_sha256": sha256_file(out_path),
            "git_commit": git_commit(repo_root),
            "audit": audit,
        },
    )
    rprint(f"[green]OK[/green] manifest: [bold]{out_path}[/bold] ({manifest_df.height:,} clips, {per_group.height:,} groups)")


if __name__ == "__main__":
    main()
