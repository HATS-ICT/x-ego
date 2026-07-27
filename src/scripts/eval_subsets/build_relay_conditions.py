"""
Assign relay conditions C1-C4 to enemy-location samples.

The question this supports: when the POV agent B cannot see an enemy, but a
teammate A can, and B can see A, does the CECL representation encode the enemy
better than baseline? At inference B's embedding comes from B's video alone, so
nothing is transmitted at test time. What is being tested is whether alignment
taught the encoder to read teammate behaviour (stance, aim direction, peek
posture) as evidence about the unseen world.

Conditions, evaluated at the prediction tick for POV agent B:

  C1  direct            B sees at least one enemy
  C2  relay + contact   B sees no enemy; some teammate A sees an enemy; B sees such an A
  C3  relay, no contact B sees no enemy; some teammate sees an enemy; B sees no such A
  C4  team blind        nobody on B's team sees any enemy

Predicted CECL gain under the shared-mental-model account is C2 > C3 > C4, with
C1 small because the information is already local. C2 versus C3 is the
discriminator. If C2 >> C3 the transfer depends on visual contact with the
informed teammate. If C2 == C3 the representation is team-aware regardless of
contact, which is a weaker relay story but still supports shared latent state.
C4 is the Area Chair's condition and isolates prior-based inference.

IMPORTANT SCOPE LIMIT. The enemy_location labels are a team-level multi-hot over
map regions for all enemies jointly, so a condition cannot be attached to one
enemy without per-place attribution. Conditions here are therefore row-level
aggregate predicates. Per-enemy detail is emitted alongside so a finer analysis
is possible later, but do not report per-enemy conditions from this file as if
the label were per-enemy.

PREREQUISITE. Requires view angles in the trajectories, which the current parse
does not extract (parse_traj_per_player.py requests only
tick/steamid/name/side/X/Y/Z/place/health with player_props=[]). Run
src/scripts/data_processing/parse_traj_with_angles.py first, then point
--trajectory-folder at its output.

Example:
    python -m src.scripts.eval_subsets.build_relay_conditions \
        --map inferno --task enemy_location_0s \
        --trajectory-folder trajectory_angles --backend geometric
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.scripts.eval_subsets.subset_common import (
    load_labels,
    load_round_trajectories,
    player_side,
    state_at_tick,
)

OPPOSITE = {"ct": "t", "t": "ct"}
EYE_HEIGHT = 64.0  # positions are at the feet; CS2 standing eye offset


# --------------------------------------------------------------------------
# Visibility backends
# --------------------------------------------------------------------------

class GeometricVisibility:
    """Field-of-view cone test. No occlusion, so this OVER-counts visibility.

    Treat results as an upper bound. A player behind a wall but inside the cone
    counts as visible, which on Inferno is a large fraction of pairs. Use this
    to prototype the pipeline and to define `sees(B, A)` for teammates, where
    same-region plus short range makes the proxy reasonable. For `sees(X, enemy)`
    prefer the awpy backend or engine spotted flags.
    """

    name = "geometric"

    def __init__(self, half_fov_deg: float = 53.0, max_dist: float = 3000.0,
                 require_same_or_missing_place: bool = False):
        self.cos_half_fov = math.cos(math.radians(half_fov_deg))
        self.max_dist = max_dist
        self.require_same_place = require_same_or_missing_place

    @staticmethod
    def _forward(pitch_deg: float, yaw_deg: float) -> np.ndarray:
        # Source-engine convention: positive pitch looks down.
        p = math.radians(float(pitch_deg))
        y = math.radians(float(yaw_deg))
        return np.array(
            [math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)],
            dtype=float,
        )

    def sees(self, observer, target) -> Optional[bool]:
        for col in ("pitch", "yaw"):
            if col not in observer or pd.isna(observer.get(col)):
                return None  # cannot evaluate without angles
        eye = np.array(
            [float(observer["X"]), float(observer["Y"]), float(observer["Z"]) + EYE_HEIGHT]
        )
        tgt = np.array(
            [float(target["X"]), float(target["Y"]), float(target["Z"]) + EYE_HEIGHT]
        )
        delta = tgt - eye
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-6 or dist > self.max_dist:
            return False
        if self.require_same_place:
            op, tp = observer.get("place"), target.get("place")
            if isinstance(op, str) and isinstance(tp, str) and op != tp:
                return False
        fwd = self._forward(observer["pitch"], observer["yaw"])
        return bool(float(np.dot(fwd, delta / dist)) >= self.cos_half_fov)


class AwpyVisibility:
    """Line-of-sight through awpy's triangle-mesh visibility checker.

    VERIFY BEFORE TRUSTING RESULTS. awpy's visibility API has changed across 2.x
    releases, so this resolves the entry point at runtime rather than hard-coding
    a signature. If construction fails it raises with the names it actually found,
    which is what you use to fix `_resolve`. Inspect with:

        python -c "import awpy.visibility as v; print(dir(v))"

    The mesh files are per-map .tri artifacts that awpy downloads separately; see
    the awpy docs for the artifact fetch step. Point --tri-path at the .tri file
    for the map if autodiscovery fails.
    """

    name = "awpy"

    def __init__(self, map_name: str, tri_path: Optional[str] = None):
        self._checker = self._resolve(map_name, tri_path)

    @staticmethod
    def _resolve(map_name: str, tri_path: Optional[str]):
        try:
            import awpy.visibility as av
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "awpy is not installed. `pip install awpy`, then fetch the map "
                "triangle artifacts per the awpy docs."
            ) from exc

        full = map_name if map_name.startswith("de_") else f"de_{map_name}"
        candidates = [n for n in dir(av) if "isib" in n and not n.startswith("_")]
        for name in candidates:
            obj = getattr(av, name)
            if not isinstance(obj, type):
                continue
            for kwargs in (
                {"path": tri_path} if tri_path else {},
                {"tri_path": tri_path} if tri_path else {},
                {"map_name": full},
                {},
            ):
                try:
                    return obj(**kwargs)
                except Exception:
                    continue
        raise RuntimeError(
            "Could not construct an awpy visibility checker. Names found in "
            f"awpy.visibility: {candidates or dir(av)}. Fix AwpyVisibility._resolve "
            "to match your awpy version, or use --backend geometric."
        )

    def sees(self, observer, target) -> Optional[bool]:
        a = (float(observer["X"]), float(observer["Y"]), float(observer["Z"]) + EYE_HEIGHT)
        b = (float(target["X"]), float(target["Y"]), float(target["Z"]) + EYE_HEIGHT)
        for meth in ("is_visible", "visible", "check_visibility"):
            fn = getattr(self._checker, meth, None)
            if callable(fn):
                res = fn(a, b)
                # Some versions return (bool, detail).
                if isinstance(res, tuple):
                    res = res[0]
                return bool(res)
        raise RuntimeError(
            "awpy checker exposes no recognised visibility method. Methods: "
            f"{[m for m in dir(self._checker) if not m.startswith('_')]}"
        )


class SpottedFlagVisibility:
    """Engine spotted state, exact and occlusion-aware, for observer->enemy only.

    Requires a boolean spotted column in the trajectories. CS2 tracks spotted
    state per enemy, not per (observer, enemy) pair in every parser build, so if
    your column is team-level this answers "does B's team see E" rather than
    "does B see E". Check which you have before assigning C2 versus C3, because
    that distinction is the whole experiment.
    """

    name = "spotted"

    def __init__(self, column: str = "is_spotted"):
        self.column = column

    def sees(self, observer, target) -> Optional[bool]:
        val = target.get(self.column)
        if val is None or pd.isna(val):
            return None
        return bool(val)


def make_backend(args, map_name: str):
    if args.backend == "geometric":
        return GeometricVisibility(
            half_fov_deg=args.half_fov, max_dist=args.max_dist
        )
    if args.backend == "awpy":
        return AwpyVisibility(map_name, args.tri_path)
    if args.backend == "spotted":
        return SpottedFlagVisibility(args.spotted_column)
    raise ValueError(args.backend)


# --------------------------------------------------------------------------
# Condition assignment
# --------------------------------------------------------------------------

def assign_condition(
    pov_row,
    teammate_rows: Dict[str, object],
    enemy_rows: Dict[str, object],
    enemy_vis,
    teammate_vis,
) -> dict:
    """Row-level condition from aggregate visibility predicates."""
    pov_sees_enemy = False
    informed_mates: list[str] = []
    unresolved = 0

    for erow in enemy_rows.values():
        v = enemy_vis.sees(pov_row, erow)
        if v is None:
            unresolved += 1
        elif v:
            pov_sees_enemy = True

    for sid, arow in teammate_rows.items():
        for erow in enemy_rows.values():
            v = enemy_vis.sees(arow, erow)
            if v:
                informed_mates.append(sid)
                break

    contact_with_informed = False
    for sid in informed_mates:
        v = teammate_vis.sees(pov_row, teammate_rows[sid])
        if v:
            contact_with_informed = True
            break

    if pov_sees_enemy:
        cond = "C1"
    elif informed_mates and contact_with_informed:
        cond = "C2"
    elif informed_mates:
        cond = "C3"
    else:
        cond = "C4"

    return {
        "condition": cond,
        "pov_sees_enemy": int(pov_sees_enemy),
        "n_informed_teammates": len(informed_mates),
        "contact_with_informed": int(contact_with_informed),
        "n_enemies_alive": len(enemy_rows),
        "n_teammates_alive": len(teammate_rows),
        "n_unresolved_enemy_pairs": unresolved,
    }


def build(args) -> pd.DataFrame:
    data_dir = str(Path(args.data_dir).resolve())
    labels = load_labels(Path(args.data_dir), args.map, args.task)
    enemy_vis = make_backend(args, args.map)
    # Teammate contact always uses geometry; no engine flag covers teammates.
    teammate_vis = GeometricVisibility(
        half_fov_deg=args.half_fov, max_dist=args.teammate_max_dist
    )

    original_folder = load_round_trajectories.__wrapped__  # bypass lru_cache signature
    rows = []
    for rec in labels.to_dict("records"):
        match_id = str(rec["match_id"])
        round_num = int(rec["round_num"])
        traj = original_folder(
            data_dir, args.map, match_id, round_num
        ) if args.trajectory_folder == "trajectory" else _load_alt(
            data_dir, args.map, args.trajectory_folder, match_id, round_num
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

        teammate_rows, enemy_rows = {}, {}
        for sid, df in traj.items():
            if sid == pov:
                continue
            side = player_side(df)
            row = state_at_tick(df, tick, require_alive=True)
            if row is None:
                continue
            if side == pov_side:
                teammate_rows[sid] = row
            elif side == enemy_side:
                enemy_rows[sid] = row

        if not enemy_rows:
            continue

        out = {"partition": rec["partition"], "idx": int(rec["idx"])}
        out.update(
            assign_condition(pov_row, teammate_rows, enemy_rows, enemy_vis, teammate_vis)
        )
        rows.append(out)

    return pd.DataFrame(rows)


def _load_alt(data_dir, map_name, folder, match_id, round_num):
    """Load trajectories from an alternate folder (e.g. the angle-augmented parse)."""
    match_dir = Path(data_dir) / map_name / folder / match_id
    out = {}
    if not match_dir.exists():
        return out
    for player_dir in match_dir.iterdir():
        if not player_dir.is_dir():
            continue
        p = player_dir / f"round_{round_num}.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not df.empty and "tick" in df.columns:
            out[player_dir.name] = df.sort_values("tick").reset_index(drop=True)
    return out


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
    ap.add_argument(
        "--backend", default="geometric", choices=["geometric", "awpy", "spotted"]
    )
    ap.add_argument("--tri-path", default=None, help="awpy backend: path to map .tri")
    ap.add_argument("--spotted-column", default="is_spotted")
    ap.add_argument("--half-fov", type=float, default=53.0)
    ap.add_argument("--max-dist", type=float, default=3000.0)
    ap.add_argument("--teammate-max-dist", type=float, default=1500.0)
    args = ap.parse_args()

    df = build(args)
    if df.empty:
        print(
            "No rows produced. Most likely the trajectory folder lacks view angles. "
            "Run parse_traj_with_angles.py first."
        )
        return

    if args.partition != "all":
        keep = {p.strip() for p in args.partition.split(",")}
        df = df[df["partition"].isin(keep)].reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.map}-{args.task}-relay-{args.backend}.csv"
    df.to_csv(out_path, index=False)

    print(f"wrote {out_path}  ({len(df)} rows)\n")
    counts = df["condition"].value_counts().sort_index()
    for cond in ["C1", "C2", "C3", "C4"]:
        n = int(counts.get(cond, 0))
        print(f"  {cond}  {n:6d} / {len(df)}  ({100.0 * n / len(df):.1f}%)")
    unresolved = int((df["n_unresolved_enemy_pairs"] > 0).sum())
    if unresolved:
        print(
            f"\nWARNING: {unresolved} rows had at least one enemy pair whose "
            "visibility could not be evaluated. Conditions for those rows are "
            "biased toward C3/C4."
        )
    if int(counts.get("C2", 0)) < 100:
        print(
            "\nWARNING: fewer than 100 C2 samples. The C2 versus C3 comparison "
            "is the point of this experiment and will not have power. Widen the "
            "partition set or pool maps before drawing conclusions."
        )


if __name__ == "__main__":
    main()
