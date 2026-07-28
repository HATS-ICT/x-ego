"""
Validate the visibility signal before building any relay condition.

Every relay number depends on two predicates being right. This script checks them
against facts that must hold regardless of what the data says, so a broken signal
is caught here rather than showing up as a plausible but meaningless C2 versus C3
comparison.

Checks, in order of how much they matter.

  1  Mask bit offset. m_bSpottedByMask is indexed by entity slot, and the offset
     from `entity_id` has varied across parser versions. Tries several offsets and
     reports which agrees best with geometry. A wrong offset is exposed by
     self-spotting, since no player is ever spotted by themselves.

  2  Sanity of the mask against the team-level flag. Whenever any bit is set in an
     enemy's mask, `spotted` must be true. A large violation rate means the two
     fields are not what we think they are.

  3  Directionality. A player's mask should name OPPONENTS. If teammate bits show
     up, the field means something other than assumed and the enemy leg of the
     relay condition is invalid.

  4  Base rates. If almost every pair is visible, or almost none, the conditions
     will not partition the data usefully no matter how correct the signal is.

Usage:
    python -m src.scripts.eval_subsets.validate_visibility --map inferno
    python -m src.scripts.eval_subsets.validate_visibility --map inferno \
        --tri-path data/inferno/mesh/de_inferno.tri
"""

from __future__ import annotations

import argparse
import collections
import random
from pathlib import Path

import pandas as pd

from src.scripts.eval_subsets.relay_io import load_alt_trajectories
from src.scripts.eval_subsets.subset_common import load_labels, player_side, state_at_tick
from src.scripts.eval_subsets.visibility import (
    MASK_COLUMN,
    FovConeVisibility,
    LineOfSightVisibility,
    SpottedMaskVisibility,
    TeamSpottedVisibility,
    decode_mask,
    resolve_mask_offset,
)
from src.utils.env_utils import resolve_data_dir

OPPOSITE = {"ct": "t", "t": "ct"}


def collect_pairs(args):
    """Sample (observer, target, relation) triples at label prediction ticks."""
    data_dir = str(Path(args.data_dir).resolve())
    labels = load_labels(Path(args.data_dir), args.map, args.task)
    if args.partition != "all":
        keep = {p.strip() for p in args.partition.split(",")}
        labels = labels[labels["partition"].isin(keep)]
    recs = labels.to_dict("records")
    random.Random(0).shuffle(recs)

    enemy_pairs, mate_pairs, self_pairs = [], [], []
    rounds_seen = 0
    for rec in recs:
        if rounds_seen >= args.max_rounds:
            break
        traj = load_alt_trajectories(
            data_dir, args.map, args.trajectory_folder,
            str(rec["match_id"]), int(rec["round_num"]),
        )
        if not traj:
            continue
        pov = str(rec["pov_steamid"])
        pov_side = str(rec["pov_side"]).lower()
        enemy_side = OPPOSITE.get(pov_side)
        tick = float(rec["prediction_tick"])

        pov_row = state_at_tick(traj.get(pov), tick, require_alive=True)
        if pov_row is None:
            continue
        if MASK_COLUMN not in pov_row.index:
            raise SystemExit(
                f"trajectories under {args.trajectory_folder} have no "
                f"{MASK_COLUMN!r} column. Re-parse with\n"
                "  python -m src.scripts.data_processing.parse_traj_with_angles "
                f"--map {args.map}"
            )
        rounds_seen += 1

        for sid, df in traj.items():
            row = state_at_tick(df, tick, require_alive=True)
            if row is None:
                continue
            if sid == pov:
                self_pairs.append((pov_row, row))
                continue
            side = player_side(df)
            if side == enemy_side:
                enemy_pairs.append((pov_row, row))
            elif side == pov_side:
                mate_pairs.append((pov_row, row))
    return enemy_pairs, mate_pairs, self_pairs, rounds_seen


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--map", required=True)
    ap.add_argument("--task", default="enemy_location_0s")
    ap.add_argument("--data-dir", default=None,
                    help='defaults to $DATA_BASE_PATH from .env, else "data"')
    ap.add_argument("--trajectory-folder", default="trajectory_angles")
    ap.add_argument("--partition", default="test")
    ap.add_argument("--tri-path", default=None,
                    help="map .tri mesh; without it geometry is FOV-cone only")
    ap.add_argument("--max-rounds", type=int, default=200)
    ap.add_argument("--half-fov", type=float, default=53.0)
    ap.add_argument("--max-dist", type=float, default=3000.0)
    args = ap.parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)

    enemy_pairs, mate_pairs, self_pairs, n_rounds = collect_pairs(args)
    print(f"sampled {n_rounds} label rows: {len(enemy_pairs)} pov-enemy pairs, "
          f"{len(mate_pairs)} pov-teammate pairs\n")
    if not enemy_pairs:
        raise SystemExit("no pairs collected; check --map, --task, --trajectory-folder")

    if args.tri_path:
        geom = LineOfSightVisibility(args.tri_path, args.half_fov, args.max_dist)
        print(f"geometry reference: line of sight via {args.tri_path}")
    else:
        geom = FovConeVisibility(args.half_fov, args.max_dist)
        print("geometry reference: FOV cone only, which over-counts. Pass --tri-path "
              "for a real occlusion test.")

    # ---- check 1, bit offset --------------------------------------------
    print("\n[1] mask bit offset")
    res = resolve_mask_offset(enemy_pairs + self_pairs, range(-2, 3), reference=geom)
    print(f"  {res['n_pairs']} pairs considered")
    for c in res["candidates"]:
        flag = "  <- self-spotting, impossible" if c["self_spotted_violations"] else ""
        print(f"    offset {c['offset']:+d}  n={c['n_evaluated']:6d}  "
              f"mask_true {c['frac_mask_true']:.3f}  "
              f"agree {c['agreement_with_reference']:.3f}{flag}")
    if "best_offset" in res:
        print(f"  best offset {res['best_offset']:+d} at agreement "
              f"{res['best_agreement']:.3f}, margin {res['margin']:.3f}")
        if res["margin"] < 0.05:
            print("  WARNING: margin under 0.05. The offset is not identifiable from "
                  "this sample,\n  so the mask cannot be trusted. Use --enemy-backend "
                  "los instead.")
        offset = res["best_offset"]
    else:
        print("  WARNING: no offset survived. Every candidate either evaluated "
              "nothing or implied\n  self-spotting. Do not use the mask backend.")
        offset = 0

    mask = SpottedMaskVisibility(bit_offset=offset)
    team = TeamSpottedVisibility()

    # ---- check 2, mask versus team flag ---------------------------------
    print("\n[2] mask implies the team-level spotted flag")
    viol = checked = 0
    for obs, tgt in enemy_pairs:
        bits = decode_mask(tgt.get(MASK_COLUMN))
        tv = team.sees(obs, tgt)
        if bits is None or tv is None:
            continue
        checked += 1
        if bits and not tv:
            viol += 1
    if checked:
        print(f"  {checked} enemies checked, {viol} with bits set but spotted=False "
              f"({100.0 * viol / checked:.2f}%)")
        if viol / checked > 0.05:
            print("  WARNING: above 5%. The two fields disagree more than rounding "
                  "explains.")
    else:
        print("  skipped, no enemy carried both fields")

    # ---- check 3, directionality ----------------------------------------
    print("\n[3] masks name opponents, not teammates")
    ent_side = {}
    for obs, tgt in enemy_pairs + mate_pairs:
        for row in (obs, tgt):
            e = row.get("entity_id")
            if e is not None and not pd.isna(e):
                ent_side[int(e) + offset] = str(row.get("side", "")).lower()
    cross = same = 0
    for _, tgt in enemy_pairs:
        bits = decode_mask(tgt.get(MASK_COLUMN)) or set()
        tside = str(tgt.get("side", "")).lower()
        for b in bits:
            s = ent_side.get(b)
            if s is None:
                continue
            if s == tside:
                same += 1
            else:
                cross += 1
    tot = same + cross
    if tot:
        print(f"  {tot} resolved bits: {cross} opponent ({100.0*cross/tot:.1f}%), "
              f"{same} same-team ({100.0*same/tot:.1f}%)")
        if same / tot > 0.10:
            print("  WARNING: over 10% of set bits name teammates. The field does not "
                  "mean what\n  the relay condition assumes. Use --enemy-backend los.")
    else:
        print("  skipped, no bits resolved to a known entity")

    # ---- check 4, base rates --------------------------------------------
    print("\n[4] base rates, do the conditions partition anything")
    for label, pairs, backends in (
        ("pov sees enemy", enemy_pairs, {"mask": mask, "geom": geom}),
        ("pov sees teammate", mate_pairs, {"geom": geom}),
    ):
        for bname, b in backends.items():
            vals = [b.sees(o, t) for o, t in pairs]
            known = [v for v in vals if v is not None]
            if not known:
                print(f"  {label:20s} {bname:5s}  no evaluable pairs")
                continue
            frac = sum(known) / len(known)
            print(f"  {label:20s} {bname:5s}  visible {frac:.3f}  "
                  f"unknown {100.0*(len(vals)-len(known))/len(vals):.1f}%")
            if bname == "geom" and frac > 0.7:
                print("      note: high, as expected when the reference is a cone "
                      "without occlusion")

    print("\nUse the best offset above with --mask-bit-offset when building "
          "conditions.\nIf checks 1 to 3 warned, prefer --enemy-backend los.")


if __name__ == "__main__":
    main()
