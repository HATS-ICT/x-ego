"""
Visibility backends for the relay conditions.

Three sources of truth, in decreasing order of trustworthiness.

  SpottedMaskVisibility   The engine's own m_bSpottedByMask, exposed by
                          demoparser2 as `approximate_spotted_by`. A bitmask on
                          each player naming which players have spotted them.
                          Occlusion-aware, smoke-aware, exactly what the game
                          decided. Only defined observer -> enemy, see below.

  LineOfSightVisibility   awpy's VisibilityChecker against the map collision
                          mesh, optionally intersected with a field-of-view cone.
                          Geometric truth about walls, but it does not know about
                          smoke, flashes, or the engine's own spotting rules.

  FovConeVisibility       Eye angles only. Ignores walls, so it OVER-counts
                          badly. A fallback for prototyping, not for results.

WHY TWO ARE NEEDED. The engine tracks spotted state for enemies. A player's mask
names the opponents who see them; teammates are on the radar regardless of line of
sight and so are absent from it. The relay condition needs both legs:

    does teammate A see enemy E     -> mask, exact
    does POV agent B see teammate A -> geometry, since the mask cannot answer it

So the builder uses SpottedMaskVisibility for the enemy leg and
LineOfSightVisibility for the teammate leg. Mixing backends across the two legs is
deliberate, not an oversight.

BIT MAPPING IS NOT ASSUMED. m_bSpottedByMask is indexed by entity slot, not by
steamid, and the offset between `entity_id` and the bit index has varied. Rather
than hard-code it, `resolve_mask_offset` finds the offset whose implied visibility
best agrees with geometry, and reports the agreement so a wrong mapping is visible
instead of silent. Run validate_visibility.py before trusting any condition counts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Positions are at the feet. CS2 standing eye offset; ducking is ~46 but the
# trajectories do not record stance, so this over-estimates for crouched players.
EYE_HEIGHT = 64.0

MASK_COLUMN = "approximate_spotted_by"
ENTITY_COLUMN = "entity_id"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _eye(row) -> np.ndarray:
    return np.array(
        [float(row["X"]), float(row["Y"]), float(row["Z"]) + EYE_HEIGHT], dtype=float
    )


def forward_vector(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Source-engine eye direction. Positive pitch looks DOWN."""
    p = math.radians(float(pitch_deg))
    y = math.radians(float(yaw_deg))
    return np.array(
        [math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)], dtype=float
    )


def decode_mask(value) -> Optional[set[int]]:
    """Bit indices set in a spotted mask, tolerating the shapes parsers emit.

    Seen in the wild: a plain int, a list of ints forming a wider mask, and a
    stringified int. Returns None when the value is absent or unparseable, which
    the callers treat as "unknown" rather than "not visible".
    """
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        bits: set[int] = set()
        for word_i, word in enumerate(value):
            try:
                w = int(word)
            except (TypeError, ValueError):
                continue
            for b in range(32):
                if w & (1 << b):
                    bits.add(word_i * 32 + b)
        return bits
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return None
    if v < 0:  # signed overflow of a wide mask
        v &= (1 << 64) - 1
    return {b for b in range(64) if v & (1 << b)}


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

class FovConeVisibility:
    """Field-of-view cone. No occlusion, so this OVER-counts. Prototyping only."""

    name = "fov"
    needs_angles = True

    def __init__(self, half_fov_deg: float = 53.0, max_dist: float = 3000.0):
        self.cos_half_fov = math.cos(math.radians(half_fov_deg))
        self.max_dist = max_dist

    def sees(self, observer, target) -> Optional[bool]:
        for col in ("pitch", "yaw"):
            if col not in observer or pd.isna(observer.get(col)):
                return None
        eye, tgt = _eye(observer), _eye(target)
        delta = tgt - eye
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-6:
            return True
        if dist > self.max_dist:
            return False
        fwd = forward_vector(observer["pitch"], observer["yaw"])
        return bool(float(np.dot(fwd, delta / dist)) >= self.cos_half_fov)


class LineOfSightVisibility:
    """awpy triangle-mesh line of sight, optionally intersected with the FOV cone.

    Requires a per-map .tri file. Build one from the map's .vphys with
    src/scripts/data_processing/build_map_tri.py, which wraps awpy's VphysParser.

    Geometry knows about walls but not about smoke, flashbangs, or the engine's
    spotting rules, so it over-counts relative to what a player actually
    registered. With require_fov it at least excludes targets behind the observer.
    """

    name = "los"
    needs_angles = False

    def __init__(self, tri_path: str | Path, half_fov_deg: float = 53.0,
                 max_dist: float = 3000.0, require_fov: bool = True):
        from awpy.visibility import VisibilityChecker

        tri_path = Path(tri_path)
        if not tri_path.exists():
            raise FileNotFoundError(
                f"no .tri mesh at {tri_path}. Build one with\n"
                f"  python -m src.scripts.data_processing.build_map_tri --map <map>"
            )
        self._checker = VisibilityChecker(path=tri_path)
        self._cone = FovConeVisibility(half_fov_deg, max_dist) if require_fov else None
        self.max_dist = max_dist
        self.require_fov = require_fov

    def sees(self, observer, target) -> Optional[bool]:
        if self._cone is not None:
            in_cone = self._cone.sees(observer, target)
            if in_cone is None:
                return None
            if not in_cone:
                return False
        a, b = _eye(observer), _eye(target)
        if float(np.linalg.norm(b - a)) > self.max_dist:
            return False
        res = self._checker.is_visible(tuple(a), tuple(b))
        if isinstance(res, tuple):  # some versions return (bool, detail)
            res = res[0]
        return bool(res)


class SpottedMaskVisibility:
    """Engine spotted state, per (observer, target) pair. Enemy targets only.

    `observer_sees_target` is true when the observer's bit is set in the target's
    m_bSpottedByMask. Because the engine only maintains this for enemies, calling
    it with a teammate target answers nothing and returns None.
    """

    name = "mask"
    needs_angles = False

    def __init__(self, bit_offset: int = 0, mask_column: str = MASK_COLUMN,
                 entity_column: str = ENTITY_COLUMN):
        self.bit_offset = bit_offset
        self.mask_column = mask_column
        self.entity_column = entity_column

    def sees(self, observer, target) -> Optional[bool]:
        bits = decode_mask(target.get(self.mask_column))
        if bits is None:
            return None
        ent = observer.get(self.entity_column)
        if ent is None or pd.isna(ent):
            return None
        return bool((int(ent) + self.bit_offset) in bits)


class TeamSpottedVisibility:
    """m_bSpotted, a team-level flag. Answers "is this enemy spotted by anyone".

    Deliberately NOT usable for the relay conditions, which turn on WHICH
    teammate sees the enemy. Kept because it is the right control for "the team
    collectively has the information" and because it validates the mask: whenever
    the mask has any bit set, this flag should be true.
    """

    name = "team_spotted"
    needs_angles = False

    def __init__(self, column: str = "spotted"):
        self.column = column

    def sees(self, observer, target) -> Optional[bool]:
        val = target.get(self.column)
        if val is None or pd.isna(val):
            return None
        return bool(val)


# --------------------------------------------------------------------------
# bit-offset resolution
# --------------------------------------------------------------------------

def resolve_mask_offset(
    pairs: Iterable[tuple], candidate_offsets: Iterable[int] = range(-2, 3),
    reference=None,
) -> dict:
    """Pick the entity-to-bit offset whose mask best agrees with geometry.

    `pairs` yields (observer_row, enemy_row). For each candidate offset the mask
    verdict is compared against `reference` (a geometric backend). The right
    offset should agree substantially; a wrong one is near chance and, more
    tellingly, will mark players as spotted by their own teammates.

    Returns every candidate's stats so the margin between the best and the rest is
    visible. A narrow margin means do not trust the mask.
    """
    pairs = list(pairs)
    out = {"n_pairs": len(pairs), "candidates": []}
    if not pairs:
        return out

    for off in candidate_offsets:
        mask = SpottedMaskVisibility(bit_offset=off)
        agree = total = mask_true = 0
        self_spotted = 0
        for obs, tgt in pairs:
            mv = mask.sees(obs, tgt)
            if mv is None:
                continue
            mask_true += int(mv)
            # A player must never be spotted by themselves.
            if str(obs.get("steamid")) == str(tgt.get("steamid")) and mv:
                self_spotted += 1
            if reference is None:
                total += 1
                continue
            rv = reference.sees(obs, tgt)
            if rv is None:
                continue
            total += 1
            agree += int(mv == rv)
        out["candidates"].append({
            "offset": off,
            "n_evaluated": total,
            "frac_mask_true": (mask_true / total) if total else float("nan"),
            "agreement_with_reference": (agree / total) if total and reference else float("nan"),
            "self_spotted_violations": self_spotted,
        })

    valid = [c for c in out["candidates"]
             if c["n_evaluated"] and not c["self_spotted_violations"]]
    if valid and reference is not None:
        best = max(valid, key=lambda c: c["agreement_with_reference"])
        out["best_offset"] = best["offset"]
        out["best_agreement"] = best["agreement_with_reference"]
        others = [c["agreement_with_reference"] for c in valid if c is not best]
        out["margin"] = best["agreement_with_reference"] - max(others) if others else float("nan")
    return out


def make_backend(kind: str, *, map_name: str | None = None, tri_path: str | None = None,
                 half_fov_deg: float = 53.0, max_dist: float = 3000.0,
                 bit_offset: int = 0, require_fov: bool = True):
    if kind == "mask":
        return SpottedMaskVisibility(bit_offset=bit_offset)
    if kind == "los":
        if tri_path is None:
            raise ValueError("the los backend needs --tri-path")
        return LineOfSightVisibility(tri_path, half_fov_deg, max_dist, require_fov)
    if kind == "fov":
        return FovConeVisibility(half_fov_deg, max_dist)
    if kind == "team_spotted":
        return TeamSpottedVisibility()
    raise ValueError(f"unknown visibility backend {kind!r}")


BACKENDS = ("mask", "los", "fov", "team_spotted")
