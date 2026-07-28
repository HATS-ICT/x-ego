"""
Assign relay conditions C1-C4 to enemy-location samples.

The question. When the POV agent B cannot see an enemy, but a teammate A can, and
B can see A, does the CECL representation encode that enemy better than baseline?
At inference B's embedding comes from B's own video alone, so nothing is
transmitted at test time. What is tested is whether cross-ego alignment taught the
encoder to read teammate behaviour, meaning stance, aim direction, peek posture, as
evidence about a part of the world B cannot see.

Conditions, evaluated at the prediction tick for POV agent B:

  C1  direct            B sees at least one enemy
  C2  relay + contact   B sees no enemy; some teammate A sees an enemy; B sees such an A
  C3  relay, no contact B sees no enemy; some teammate sees an enemy; B sees no such A
  C4  team blind        nobody on B's team sees any enemy

Predicted CECL gain under the shared-mental-model reading is C2 > C3 > C4, with C1
small because the information is already local. C2 versus C3 is the discriminator.
If C2 greatly exceeds C3 the transfer depends on visual contact with the informed
teammate. If they are equal the representation is team-aware regardless of contact,
a weaker relay story that still supports shared latent state. C4 isolates
prior-based inference and is the Area Chair's condition.

TWO BACKENDS, ON PURPOSE. The engine maintains spotted state for enemies only, so
m_bSpottedByMask answers "does teammate A see enemy E" exactly but says nothing
about "does B see teammate A". The enemy leg therefore defaults to the mask and the
teammate leg to geometry. See visibility.py.

SCOPE LIMIT. enemy_location labels are a team-level multi-hot over map regions for
all enemies jointly, so a condition cannot be attached to one enemy without
per-region attribution. Conditions here are row-level aggregate predicates. Where
a region has exactly one occupant, attribution is possible and the columns
`n_regions_single_occupant` and `relay_regions` record it, which supports a finer
label-level analysis later. Do not report per-enemy conditions from the aggregate
columns.

RUN validate_visibility.py FIRST. The mask bit offset is not fixed across parser
versions and a wrong offset produces plausible, meaningless counts.

Example:
    python -m src.scripts.eval_subsets.build_relay_conditions \
        --map inferno --task enemy_location_0s \
        --tri-path data/inferno/mesh/de_inferno.tri --mask-bit-offset 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.scripts.eval_subsets.relay_io import load_alt_trajectories
from src.scripts.eval_subsets.subset_common import (
    load_labels,
    place_to_idx_for_map,
    player_side,
    state_at_tick,
)
from src.scripts.eval_subsets.visibility import (
    MASK_COLUMN,
    FovConeVisibility,
    LineOfSightVisibility,
    SpottedMaskVisibility,
    TeamSpottedVisibility,
)

OPPOSITE = {"ct": "t", "t": "ct"}


def _make(kind: str, args, label: str):
    if kind == "mask":
        return SpottedMaskVisibility(bit_offset=args.mask_bit_offset)
    if kind == "team_spotted":
        return TeamSpottedVisibility()
    if kind == "los":
        if not args.tri_path:
            raise SystemExit(
                f"--{label}-backend los needs --tri-path. Build the mesh with\n"
                f"  python -m src.scripts.data_processing.build_map_tri --map {args.map}\n"
                f"or use --{label}-backend fov, which ignores walls and over-counts."
            )
        return LineOfSightVisibility(
            args.tri_path, args.half_fov, args.max_dist, require_fov=not args.no_fov
        )
    if kind == "fov":
        return FovConeVisibility(args.half_fov, args.max_dist)
    raise ValueError(kind)


def assign_condition(
    pov_row,
    teammate_rows: Dict[str, object],
    enemy_rows: Dict[str, object],
    enemy_vis,
    teammate_vis,
    place_to_idx: Optional[dict] = None,
) -> dict:
    """Row-level condition plus the per-region detail needed for attribution."""
    pov_sees_enemy = False
    unresolved_enemy = 0
    seen_by_pov: set[str] = set()

    for sid, erow in enemy_rows.items():
        v = enemy_vis.sees(pov_row, erow)
        if v is None:
            unresolved_enemy += 1
        elif v:
            pov_sees_enemy = True
            seen_by_pov.add(sid)

    # Which teammates see an enemy the POV agent does not.
    informed: dict[str, set[str]] = {}
    for sid, arow in teammate_rows.items():
        seen = set()
        for esid, erow in enemy_rows.items():
            if enemy_vis.sees(arow, erow):
                seen.add(esid)
        exclusive = seen - seen_by_pov
        if exclusive:
            informed[sid] = exclusive

    contact = [sid for sid in informed if teammate_vis.sees(pov_row, teammate_rows[sid])]

    if pov_sees_enemy:
        cond = "C1"
    elif informed and contact:
        cond = "C2"
    elif informed:
        cond = "C3"
    else:
        cond = "C4"

    # Regions holding exactly one enemy can be attributed to that enemy, which
    # turns the team-level multi-hot into a per-enemy question for those labels.
    occupancy: dict[str, list[str]] = {}
    for sid, erow in enemy_rows.items():
        p = erow.get("place")
        if isinstance(p, str) and p:
            occupancy.setdefault(p, []).append(sid)
    single = {p: sids[0] for p, sids in occupancy.items() if len(sids) == 1}
    relayed = {p for p, sid in single.items()
               if sid not in seen_by_pov
               and any(sid in ex for ex in (informed.get(m, set()) for m in contact))}
    relay_idx = sorted(
        place_to_idx[p] for p in relayed
        if place_to_idx and p in place_to_idx
    ) if place_to_idx else []

    return {
        "condition": cond,
        "pov_sees_enemy": int(pov_sees_enemy),
        "n_enemies_seen_by_pov": len(seen_by_pov),
        "n_informed_teammates": len(informed),
        "n_informed_with_contact": len(contact),
        "contact_with_informed": int(bool(contact)),
        "n_enemies_alive": len(enemy_rows),
        "n_teammates_alive": len(teammate_rows),
        "n_unresolved_enemy_pairs": unresolved_enemy,
        "n_regions_single_occupant": len(single),
        # Label columns attributable to an enemy visible only via a teammate the
        # POV agent can see. Empty for most rows; the basis of the label-level test.
        "relay_regions": ";".join(str(i) for i in relay_idx),
        "n_relay_regions": len(relay_idx),
    }


def build(args) -> pd.DataFrame:
    data_dir = str(Path(args.data_dir).resolve())
    labels = load_labels(Path(args.data_dir), args.map, args.task)
    enemy_vis = _make(args.enemy_backend, args, "enemy")
    teammate_vis = _make(args.teammate_backend, args, "teammate")
    try:
        p2i = place_to_idx_for_map(args.map)
    except Exception:
        p2i = None

    rows, checked_columns = [], False
    for rec in labels.to_dict("records"):
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

        if not checked_columns:
            checked_columns = True
            need = {"pitch", "yaw"}
            if args.enemy_backend == "mask":
                need |= {MASK_COLUMN, "entity_id"}
            absent = sorted(need - set(pov_row.index))
            if absent:
                raise SystemExit(
                    f"trajectories under {args.trajectory_folder!r} lack {absent}.\n"
                    "Re-parse with\n"
                    "  python -m src.scripts.data_processing.parse_traj_with_angles "
                    f"--map {args.map}"
                )

        teammate_rows, enemy_rows = {}, {}
        for sid, df in traj.items():
            if sid == pov:
                continue
            row = state_at_tick(df, tick, require_alive=True)
            if row is None:
                continue
            side = player_side(df)
            if side == pov_side:
                teammate_rows[sid] = row
            elif side == enemy_side:
                enemy_rows[sid] = row

        if not enemy_rows:
            continue

        out = {"partition": rec["partition"], "idx": int(rec["idx"])}
        out.update(assign_condition(
            pov_row, teammate_rows, enemy_rows, enemy_vis, teammate_vis, p2i
        ))
        rows.append(out)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--map", required=True)
    ap.add_argument("--task", default="enemy_location_0s")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--trajectory-folder", default="trajectory_angles")
    ap.add_argument("--out-dir", default="output/eval_subsets")
    ap.add_argument("--partition", default="test")
    ap.add_argument("--enemy-backend", default="mask",
                    choices=["mask", "los", "fov", "team_spotted"],
                    help="observer -> enemy. mask is the engine's own answer")
    ap.add_argument("--teammate-backend", default="los",
                    choices=["los", "fov"],
                    help="observer -> teammate. The mask cannot answer this")
    ap.add_argument("--tri-path", default=None, help="map .tri mesh for the los backend")
    ap.add_argument("--mask-bit-offset", type=int, default=0,
                    help="entity_id to mask-bit offset; get it from validate_visibility")
    ap.add_argument("--half-fov", type=float, default=53.0)
    ap.add_argument("--max-dist", type=float, default=3000.0)
    ap.add_argument("--no-fov", action="store_true",
                    help="los backend: skip the cone test, keeping pure line of sight")
    args = ap.parse_args()

    df = build(args)
    if df.empty:
        print("No rows produced. Check --map, --task, and --trajectory-folder.")
        return

    if args.partition != "all":
        keep = {p.strip() for p in args.partition.split(",")}
        df = df[df["partition"].isin(keep)].reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.enemy_backend}-{args.teammate_backend}"
    out_path = out_dir / f"{args.map}-{args.task}-relay-{tag}.csv"
    df.to_csv(out_path, index=False)

    print(f"wrote {out_path}  ({len(df)} rows)\n")
    counts = df["condition"].value_counts()
    for cond, desc in (("C1", "B sees an enemy"),
                       ("C2", "relay with contact"),
                       ("C3", "relay without contact"),
                       ("C4", "team blind")):
        n = int(counts.get(cond, 0))
        print(f"  {cond}  {n:6d} / {len(df)}  ({100.0 * n / len(df):5.1f}%)  {desc}")

    attributable = int((df["n_relay_regions"] > 0).sum())
    print(f"\n  rows with an attributable relay region: {attributable} "
          f"({100.0 * attributable / len(df):.1f}%)")

    unresolved = int((df["n_unresolved_enemy_pairs"] > 0).sum())
    if unresolved:
        print(f"\nWARNING: {unresolved} rows ({100.0*unresolved/len(df):.1f}%) had an "
              "enemy pair whose visibility\n  could not be evaluated. Those rows are "
              "biased toward C3 and C4.")
    for cond in ("C2", "C3"):
        if int(counts.get(cond, 0)) < 100:
            print(f"\nWARNING: only {int(counts.get(cond, 0))} {cond} samples. C2 versus "
                  "C3 is the point of this\n  experiment and will not have power. Pool "
                  "maps or widen --partition before\n  drawing any conclusion.")
            break


if __name__ == "__main__":
    main()
