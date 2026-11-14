#!/usr/bin/env python3
"""
P0 -- freeze an auditable match-held-out protocol.

The historical partition column is assigned per match-round, so every match
appears in train, val and test at once. This script assigns *complete matches*
to train/val/test, rewrites the contrastive CSV and every downstream task CSV
under a new cohort name, runs the overlap audits, and records an immutable
manifest (rule, seed, counts, file hashes, source hashes, git commit).

Nothing existing is overwritten: new files are written next to the originals
under cohort-suffixed names.

Usage:
    uv run python -m src.scripts.iclr2027.build_match_heldout_split --map mirage
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import polars as pl
from dotenv import load_dotenv
from rich import print as rprint

from .split_common import (
    COHORT,
    PARTITIONS,
    SPLIT_RATIOS,
    SPLIT_SEED,
    cohort_dir,
    contrastive_filename,
    git_commit,
    sha256_file,
    tasks_folder,
    write_manifest,
)

# Assignment rule, recorded verbatim in the manifest so it can be replayed.
SPLIT_RULE = (
    "Matches are the grouping unit. Matches are sorted by descending clip count "
    "(ties broken by a seeded permutation of match_id), then assigned greedily to "
    "the partition with the largest remaining clip deficit relative to "
    "SPLIT_RATIOS. Every round, player view and overlapping window of a match "
    "lands in the same partition."
)


def assign_matches(clip_counts: Dict[str, int], seed: int) -> Dict[str, str]:
    """Greedy, deterministic, clip-count-balanced match assignment."""
    import random

    rng = random.Random(seed)
    matches = sorted(clip_counts)
    tiebreak = {m: rng.random() for m in matches}
    ordered = sorted(matches, key=lambda m: (-clip_counts[m], tiebreak[m]))

    total = sum(clip_counts.values())
    targets = {p: SPLIT_RATIOS[p] * total for p in PARTITIONS}
    current = {p: 0 for p in PARTITIONS}
    assignment: Dict[str, str] = {}

    for match in ordered:
        # Largest absolute deficit wins; ties resolved by the fixed partition order.
        best = max(PARTITIONS, key=lambda p: (targets[p] - current[p], -PARTITIONS.index(p)))
        assignment[match] = best
        current[best] += clip_counts[match]

    return assignment


def audit_overlap(df: pl.DataFrame, name: str, window_cols: List[str]) -> Dict:
    """Check match / round / window disjointness and report player overlap."""
    report: Dict = {"file": name, "rows": len(df)}

    def keyset(partition: str, cols: List[str]) -> set:
        sub = df.filter(pl.col("partition") == partition).select(cols).unique()
        return set(map(tuple, sub.rows()))

    for label, cols in (
        ("match", ["match_id"]),
        ("match_round", ["match_id", "round_num"]),
        ("window", window_cols),
    ):
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        sets = {p: keyset(p, cols) for p in PARTITIONS}
        report[f"{label}_counts"] = {p: len(sets[p]) for p in PARTITIONS}
        report[f"{label}_overlap"] = {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        }

    # Player overlap is reported, not enforced: match-held-out is not player-held-out.
    player_col = next((c for c in ("pov_steamid", "teammate_0_id") if c in df.columns), None)
    if player_col is not None:
        sets = {p: keyset(p, [player_col]) for p in PARTITIONS}
        report["player_col"] = player_col
        report["player_counts"] = {p: len(sets[p]) for p in PARTITIONS}
        report["player_overlap"] = {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        }

    report["rows_per_partition"] = {
        p: int(df.filter(pl.col("partition") == p).height) for p in PARTITIONS
    }
    return report


def remap(df: pl.DataFrame, assignment: Dict[str, str]) -> pl.DataFrame:
    mapping = pl.DataFrame(
        {"match_id": list(assignment), "__new_partition": [assignment[m] for m in assignment]}
    )
    out = df.join(mapping, on="match_id", how="left")
    unmapped = out.filter(pl.col("__new_partition").is_null()).height
    if unmapped:
        raise ValueError(f"{unmapped} rows have a match_id absent from the split assignment")
    return out.drop("partition").rename({"__new_partition": "partition"}).select(df.columns)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build the match-held-out cohort split (P0)")
    parser.add_argument("--map", default="mirage")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--dry-run", action="store_true", help="audit only, write nothing")
    args = parser.parse_args()

    repo_root = Path(os.getenv("SRC_BASE_PATH", ".")).resolve()
    data_root = Path(os.environ["DATA_BASE_PATH"])
    map_dir = data_root / args.map

    src_contrastive = map_dir / "labels" / "contrastive.csv"
    src_tasks_dir = map_dir / "labels" / "all_tasks"
    out_dir = cohort_dir(data_root, args.map)
    out_tasks_dir = map_dir / "labels" / tasks_folder()

    contrastive = pl.read_csv(src_contrastive, null_values=[])
    clip_counts = dict(
        contrastive.group_by("match_id")
        .agg(pl.col("num_alive_teammates").sum().alias("clips"))
        .iter_rows()
    )
    rprint(f"[blue]->[/blue] {len(clip_counts)} matches, {sum(clip_counts.values()):,} observer clips")

    assignment = assign_matches(clip_counts, args.seed)
    per_partition = {
        p: {
            "matches": sum(1 for m in assignment if assignment[m] == p),
            "clips": sum(clip_counts[m] for m in assignment if assignment[m] == p),
        }
        for p in PARTITIONS
    }
    for p in PARTITIONS:
        share = per_partition[p]["clips"] / sum(clip_counts.values())
        rprint(
            f"  [cyan]{p:5s}[/cyan] matches={per_partition[p]['matches']:3d} "
            f"clips={per_partition[p]['clips']:6,d} ({share:.1%}, target {SPLIT_RATIOS[p]:.0%})"
        )

    if min(per_partition[p]["matches"] for p in ("val", "test")) < 5:
        raise ValueError(
            "val/test hold fewer than 5 independent matches; expand the partition or "
            "switch to predeclared grouped folds before using this cohort"
        )

    new_contrastive = remap(contrastive, assignment)
    audits = [audit_overlap(new_contrastive, "contrastive", ["match_id", "round_num", "start_tick", "end_tick"])]

    task_files = sorted(src_tasks_dir.glob("*.csv"))
    remapped_tasks = {}
    for task_path in task_files:
        task_df = pl.read_csv(task_path, null_values=[])
        remapped = remap(task_df, assignment)
        remapped_tasks[task_path.name] = remapped
        audits.append(
            audit_overlap(remapped, task_path.name, ["match_id", "round_num", "start_tick", "end_tick"])
        )

    bad = [
        a for a in audits
        if any(v for k, val in a.items() if k.endswith("_overlap") and not k.startswith("player")
               for v in val.values())
    ]
    if bad:
        raise ValueError(f"match/round/window overlap detected in: {[a['file'] for a in bad]}")
    rprint("[green]OK[/green] no match/round/window overlap across partitions in any file")

    if args.dry_run:
        rprint("[yellow]dry-run[/yellow] nothing written")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tasks_dir.mkdir(parents=True, exist_ok=True)

    split_path = out_dir / "match_split.csv"
    pl.DataFrame(
        {
            "match_id": sorted(assignment),
            "partition": [assignment[m] for m in sorted(assignment)],
            "clips": [clip_counts[m] for m in sorted(assignment)],
        }
    ).write_csv(split_path)

    contrastive_out = map_dir / "labels" / contrastive_filename()
    new_contrastive.write_csv(contrastive_out)
    for name, frame in remapped_tasks.items():
        frame.write_csv(out_tasks_dir / name)

    written = {
        "match_split.csv": sha256_file(split_path),
        f"labels/{contrastive_filename()}": sha256_file(contrastive_out),
        **{
            f"labels/{tasks_folder()}/{name}": sha256_file(out_tasks_dir / name)
            for name in remapped_tasks
        },
    }

    write_manifest(
        out_dir / "manifest.json",
        {
            "cohort": COHORT,
            "map": args.map,
            "split_rule": SPLIT_RULE,
            "split_seed": args.seed,
            "split_ratios": SPLIT_RATIOS,
            "grouping_unit": "match_id",
            "per_partition": per_partition,
            "total_matches": len(assignment),
            "total_clips": sum(clip_counts.values()),
            "source_files": {
                "labels/contrastive.csv": sha256_file(src_contrastive),
                **{f"labels/all_tasks/{p.name}": sha256_file(p) for p in task_files},
            },
            "written_files": written,
            "git_commit": git_commit(repo_root),
            "audits": audits,
            "caveat": (
                "match-held-out is not player-held-out; see audits[*].player_overlap"
            ),
        },
    )
    rprint(f"[green]OK[/green] cohort written to [bold]{out_dir}[/bold]")
    rprint(f"[green]OK[/green] contrastive CSV: {contrastive_out}")
    rprint(f"[green]OK[/green] {len(remapped_tasks)} task CSVs: {out_tasks_dir}")


if __name__ == "__main__":
    main()
