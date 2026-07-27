"""
Shared loaders for building evaluation subsets.

An "evaluation subset" is a per-sample boolean/numeric annotation joined to an
existing task label CSV by (partition, idx). Nothing here re-trains or re-scores
anything; the output is metadata that lets you slice per-sample predictions
offline into controlled conditions.

Join key: the downstream dataloader already returns `original_csv_idx`
(src/dataset/downstream.py), which is the `idx` column of the task label CSV.
So subset flags emitted here join directly onto dumped predictions.

Layout assumed under --data-dir:
    {map}/trajectory/{match_id}/{steamid}/round_{N}.csv
    {map}/metadata/{match_id}.json
    {map}/labels/all_tasks/{task_id}.csv
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

TICK_RATE = 64


def labels_path(data_dir: Path, map_name: str, task_id: str) -> Path:
    return data_dir / map_name / "labels" / "all_tasks" / f"{task_id}.csv"


def load_labels(data_dir: Path, map_name: str, task_id: str) -> pd.DataFrame:
    path = labels_path(data_dir, map_name, task_id)
    if not path.exists():
        raise FileNotFoundError(f"No label CSV at {path}")
    return pd.read_csv(path)


@lru_cache(maxsize=64)
def load_metadata(data_dir_str: str, map_name: str, match_id: str) -> Optional[dict]:
    path = Path(data_dir_str) / map_name / "metadata" / f"{match_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_info(meta: dict, round_num: int) -> Optional[dict]:
    if not meta or "rounds" not in meta:
        return None
    return next((r for r in meta["rounds"] if r.get("round_number") == round_num), None)


@lru_cache(maxsize=4096)
def load_round_trajectories(
    data_dir_str: str, map_name: str, match_id: str, round_num: int
) -> Dict[str, pd.DataFrame]:
    """All players' trajectories for one round, keyed by steamid, sorted by tick.

    Cached because every label row in a round hits the same files.
    """
    match_dir = Path(data_dir_str) / map_name / "trajectory" / match_id
    out: Dict[str, pd.DataFrame] = {}
    if not match_dir.exists():
        return out
    for player_dir in match_dir.iterdir():
        if not player_dir.is_dir():
            continue
        csv_path = player_dir / f"round_{round_num}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty or "tick" not in df.columns:
            continue
        out[player_dir.name] = df.sort_values("tick").reset_index(drop=True)
    return out


def state_at_tick(df: pd.DataFrame, tick: float, require_alive: bool = True):
    """Row at the latest observed tick <= `tick`, or None.

    Returns None when the player has no row at or before `tick` (not yet
    recorded) or when the last such row is beyond the player's coverage, which
    in these trajectories means dead. `require_alive` additionally drops rows
    with health <= 0.
    """
    if df is None or df.empty:
        return None
    ticks = df["tick"].to_numpy()
    if tick < ticks[0] or tick > ticks[-1]:
        return None
    pos = int(np.searchsorted(ticks, tick, side="right")) - 1
    if pos < 0:
        return None
    row = df.iloc[pos]
    if require_alive and "health" in df.columns and float(row["health"]) <= 0:
        return None
    return row


def player_side(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty or "side" not in df.columns:
        return None
    return str(df.iloc[0]["side"]).lower()


def target_steamids(
    trajectories: Dict[str, pd.DataFrame],
    pov_steamid: str,
    pov_side: str,
    group: str,
) -> list[str]:
    """`group` is 'enemy' (opposite side) or 'teammate' (same side, excluding POV)."""
    pov_side = str(pov_side).lower()
    out = []
    for sid, df in trajectories.items():
        side = player_side(df)
        if side is None:
            continue
        if group == "enemy" and side != pov_side:
            out.append(sid)
        elif group == "teammate" and side == pov_side and str(sid) != str(pov_steamid):
            out.append(sid)
    return sorted(out)


def places_at_tick(
    trajectories: Dict[str, pd.DataFrame], steamids: list[str], tick: float
) -> set[str]:
    """Set of `place` values occupied by the given living players at `tick`."""
    places = set()
    for sid in steamids:
        row = state_at_tick(trajectories.get(sid), tick, require_alive=True)
        if row is None:
            continue
        place = row.get("place")
        if isinstance(place, str) and place:
            places.add(place)
    return places


def place_to_idx_for_map(map_name: str) -> dict:
    """Place-name to label-column index mapping used when the labels were built."""
    from src.scripts.task_creator.task_definitions import get_place_to_idx_for_map

    full = map_name if map_name.startswith("de_") else f"de_{map_name}"
    return get_place_to_idx_for_map(full)


def multihot_from_places(places: set[str], place_to_idx: dict, num_labels: int) -> np.ndarray:
    vec = np.zeros(num_labels, dtype=int)
    for p in places:
        idx = place_to_idx.get(p)
        if idx is not None and 0 <= idx < num_labels:
            vec[idx] = 1
    return vec


def label_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("label_")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))
