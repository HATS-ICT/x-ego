"""Trajectory loading for the relay pipeline.

The relay conditions read the angle-augmented parse
({map}/trajectory_angles/...) rather than the original {map}/trajectory/...,
so they need a loader that takes the folder as an argument. subset_common's
loader hard-codes "trajectory" inside an lru_cache key, and reaching around
that with __wrapped__ defeats the cache. This is the same logic with the
folder as a first-class parameter, cached on it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import pandas as pd


@lru_cache(maxsize=4096)
def load_alt_trajectories(
    data_dir_str: str, map_name: str, folder: str, match_id: str, round_num: int
) -> Dict[str, pd.DataFrame]:
    """All players' trajectories for one round from `folder`, keyed by steamid."""
    match_dir = Path(data_dir_str) / map_name / folder / match_id
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
