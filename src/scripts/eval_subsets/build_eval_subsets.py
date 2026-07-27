"""
Build per-sample evaluation subsets from data already on disk.

Three subsets, none of which need re-parsing, re-training, or new video:

  moved  For forecast location tasks. Flags samples where the target's map
         region at the prediction tick differs from its region at the end of
         the observed window. On those samples the answer provably cannot be
         copied forward from the present, which removes the autocorrelation
         explanation for a forecast gain.

  bomb   For global_bombPlanted. Flags samples where the plant occurred before
         the observation window opened. With the HUD masked, the defining event
         is then outside every teammate's input and has no symbolic trace.

  alive  For enemy_aliveCount / teammate_aliveCount. Flags samples with no
         death inside the observation window, so the count reflects kills that
         happened earlier and, with the kill feed masked, were never seen.

Output is a CSV of flags keyed by (partition, idx), joinable to dumped
per-sample predictions via `original_csv_idx`.

Examples:
    python -m src.scripts.eval_subsets.build_eval_subsets moved \
        --map inferno --task enemy_location_10s

    python -m src.scripts.eval_subsets.build_eval_subsets bomb \
        --map inferno --task global_bombPlanted

    python -m src.scripts.eval_subsets.build_eval_subsets alive \
        --map inferno --task enemy_aliveCount
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.eval_subsets.subset_common import (
    label_columns,
    load_labels,
    load_metadata,
    load_round_trajectories,
    multihot_from_places,
    place_to_idx_for_map,
    places_at_tick,
    round_info,
    target_steamids,
)

OPPOSITE = {"ct": "t", "t": "ct"}


def _group_for_task(task_id: str) -> str:
    if task_id.startswith("enemy_"):
        return "enemy"
    if task_id.startswith("teammate_"):
        return "teammate"
    raise ValueError(f"Cannot infer target group from task_id {task_id!r}")


def build_moved(args) -> pd.DataFrame:
    """Flag forecast samples where the target changed region over the horizon."""
    data_dir = str(Path(args.data_dir).resolve())
    labels = load_labels(Path(args.data_dir), args.map, args.task)
    group = _group_for_task(args.task)
    lab_cols = label_columns(labels)
    p2i = place_to_idx_for_map(args.map)

    if "horizon_sec" not in labels.columns:
        print(
            f"WARNING: {args.task} has no horizon_sec column, so it is probably a "
            "nowcast task. 'moved' is only meaningful for forecast tasks."
        )

    rows = []
    for rec in labels.to_dict("records"):
        match_id = str(rec["match_id"])
        round_num = int(rec["round_num"])
        traj = load_round_trajectories(data_dir, args.map, match_id, round_num)
        if not traj:
            continue

        targets = target_steamids(traj, str(rec["pov_steamid"]), rec["pov_side"], group)
        if not targets:
            continue

        end_tick = float(rec["end_tick"])
        pred_tick = float(rec["prediction_tick"])
        places_end = places_at_tick(traj, targets, end_tick)
        places_pred = places_at_tick(traj, targets, pred_tick)

        union = places_end | places_pred
        inter = places_end & places_pred
        jaccard = (len(inter) / len(union)) if union else np.nan

        # Self-check: the multi-hot we reconstruct at the prediction tick should
        # match the stored label. A low agreement rate means the tick lookup or
        # the target group is wrong, and every flag below is then untrustworthy.
        stored = np.array([int(rec[c]) for c in lab_cols], dtype=int)
        recomputed = multihot_from_places(places_pred, p2i, len(lab_cols))

        rows.append(
            {
                "partition": rec["partition"],
                "idx": int(rec["idx"]),
                "n_targets": len(targets),
                "n_places_end": len(places_end),
                "n_places_pred": len(places_pred),
                "jaccard": jaccard,
                "moved_any": int(places_end != places_pred),
                "moved_disjoint": int(len(inter) == 0 and bool(union)),
                "label_match": int(np.array_equal(stored, recomputed)),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        agree = out["label_match"].mean()
        print(f"label reconstruction agreement: {agree:.3f} over {len(out)} rows")
        if agree < 0.95:
            print(
                "WARNING: agreement below 0.95. Inspect tick alignment and target "
                "selection before trusting the moved flags."
            )
    return out


def build_bomb(args) -> pd.DataFrame:
    """Flag bomb-state samples whose plant precedes the observation window."""
    data_dir = str(Path(args.data_dir).resolve())
    labels = load_labels(Path(args.data_dir), args.map, args.task)

    rows = []
    for rec in labels.to_dict("records"):
        match_id = str(rec["match_id"])
        round_num = int(rec["round_num"])
        meta = load_metadata(data_dir, args.map, match_id)
        rinfo = round_info(meta, round_num) if meta else None
        if rinfo is None:
            continue

        plant = rinfo.get("bomb_plant_tick")
        plant = float(plant) if plant is not None and not pd.isna(plant) else None
        start_tick = float(rec["start_tick"])
        end_tick = float(rec["end_tick"])
        pred_tick = float(rec["prediction_tick"])

        planted_at_pred = int(plant is not None and plant <= pred_tick)
        rows.append(
            {
                "partition": rec["partition"],
                "idx": int(rec["idx"]),
                "plant_tick": plant if plant is not None else -1,
                "planted_at_pred": planted_at_pred,
                # The condition of interest: label is "planted" and the plant
                # event happened strictly before the observed window opened.
                "plant_before_window": int(planted_at_pred and plant < start_tick),
                "plant_inside_window": int(
                    plant is not None and start_tick <= plant <= end_tick
                ),
                "ticks_plant_to_window_start": (
                    (start_tick - plant) if plant is not None else np.nan
                ),
                "pov_side": rec.get("pov_side"),
                "bomb_site": rinfo.get("bomb_site"),
            }
        )
    return pd.DataFrame(rows)


def build_alive(args) -> pd.DataFrame:
    """Flag alive-count samples with no death inside the observation window."""
    data_dir = str(Path(args.data_dir).resolve())
    labels = load_labels(Path(args.data_dir), args.map, args.task)
    group = _group_for_task(args.task)

    rows = []
    for rec in labels.to_dict("records"):
        match_id = str(rec["match_id"])
        round_num = int(rec["round_num"])
        meta = load_metadata(data_dir, args.map, match_id)
        if not meta:
            continue

        pov_side = str(rec["pov_side"]).lower()
        counted_side = OPPOSITE.get(pov_side) if group == "enemy" else pov_side
        if counted_side is None:
            continue

        start_tick = float(rec["start_tick"])
        end_tick = float(rec["end_tick"])

        in_window = 0
        before_window = 0
        for kill in meta.get("kills", []):
            if kill.get("round_number") != round_num:
                continue
            if str(kill.get("victim_side", "")).lower() != counted_side:
                continue
            tick = float(kill.get("tick", -1))
            if start_tick <= tick <= end_tick:
                in_window += 1
            elif tick < start_tick:
                before_window += 1

        rows.append(
            {
                "partition": rec["partition"],
                "idx": int(rec["idx"]),
                "counted_side": counted_side,
                "deaths_in_window": in_window,
                "deaths_before_window": before_window,
                # The condition of interest: the count is non-trivial (somebody
                # already died) yet no death is observable inside the window.
                "no_death_in_window": int(in_window == 0),
                "unobservable_count": int(in_window == 0 and before_window > 0),
            }
        )
    return pd.DataFrame(rows)


BUILDERS = {"moved": build_moved, "bomb": build_bomb, "alive": build_alive}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subset", choices=sorted(BUILDERS))
    ap.add_argument("--map", required=True, help="e.g. inferno, dust2, mirage")
    ap.add_argument("--task", required=True, help="task_id, e.g. enemy_location_10s")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="output/eval_subsets")
    ap.add_argument(
        "--partition",
        default="test",
        help="comma-separated partitions to keep, or 'all'",
    )
    args = ap.parse_args()

    df = BUILDERS[args.subset](args)
    if df.empty:
        print("No rows produced. Check --map, --task, and --data-dir.")
        return

    if args.partition != "all":
        keep = {p.strip() for p in args.partition.split(",")}
        df = df[df["partition"].isin(keep)].reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.map}-{args.task}-{args.subset}.csv"
    df.to_csv(out_path, index=False)

    print(f"\nwrote {out_path}  ({len(df)} rows)")
    flag_cols = [
        c
        for c in df.columns
        if c.startswith(("moved_", "plant_", "no_death", "unobservable", "planted_"))
        and df[c].dropna().isin([0, 1]).all()
    ]
    for col in flag_cols:
        n = int(df[col].sum())
        print(f"  {col:28s} {n:6d} / {len(df)}  ({100.0 * n / len(df):.1f}%)")


if __name__ == "__main__":
    main()
