"""
Shared helpers for the ICLR-2027 follow-up cohort (P0 protocol).

Every artifact written by the scripts in this package is hashed and recorded in
a manifest so a second person can reconstruct row membership, split lineage and
seeds from the saved files alone.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import polars as pl

# Cohort-wide constants. Changing any of these means a new cohort name.
COHORT = "matchheldout_v1"
SPLIT_SEED = 20260921
SPLIT_RATIOS = {"train": 0.64, "val": 0.18, "test": 0.18}

PARTITIONS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(repo_root: Path) -> str:
    """Return the current commit plus a dirty flag; never raise."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def cohort_dir(data_root: Path, map_name: str) -> Path:
    return data_root / map_name / "splits" / COHORT


def contrastive_filename() -> str:
    return f"contrastive_{COHORT}.csv"


def tasks_folder() -> str:
    return f"all_tasks_{COHORT}"


def stable_row_id(
    match_id: str,
    round_num: int,
    observer: str,
    start_tick: int,
    end_tick: int,
    target: str,
    target_tick: int,
) -> str:
    """
    Stable identifier for a task row.

    A CSV row number alone is not sufficient after regeneration, so the ID
    carries match, round, observer, window ticks, target and target tick.
    """
    payload = f"{match_id}|{round_num}|{observer}|{start_tick}|{end_tick}|{target}|{target_tick}"
    return sha256_text(payload)[:24]


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["written_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hash_frame(df: pl.DataFrame, columns: Iterable[str]) -> str:
    """Order-independent hash of the selected columns of a frame."""
    cols = [c for c in columns if c in df.columns]
    sub = df.select(cols).sort(cols)
    return sha256_text(str(sub.to_dicts()))


TICK_RATE = 64


def window_id(match_id: str, round_num: int, observer: str, start_tick_norm: int, end_tick_norm: int) -> str:
    """
    Stable key for one observer clip (the unit that is encoded once and cached).
    Uses round-relative ticks, which is how both loaders locate video.
    """
    payload = f"{match_id}|{int(round_num)}|{observer}|{int(start_tick_norm)}|{int(end_tick_norm)}"
    return sha256_text(payload)[:20]


def ensure_tick_norm(df, data_root, map_name: str):
    """
    Add start/end_tick_norm when a task CSV only has absolute ticks.

    tick_norm = tick - freeze_end_tick of that round (the convention every other
    task CSV follows, verified against the round metadata). Returns (df, derived).
    """
    import json as _json
    from pathlib import Path as _Path

    import polars as _pl

    if "start_tick_norm" in df.columns and "end_tick_norm" in df.columns:
        return df, False
    rows = []
    for mid in df["match_id"].unique().to_list():
        meta = _json.loads((_Path(data_root) / map_name / "metadata" / f"{mid}.json").read_text())
        rows += [{"match_id": mid, "round_num": int(r["round_number"]), "__freeze": int(r["freeze_end_tick"])} for r in meta["rounds"]]
    fr = _pl.DataFrame(rows)
    out = df.with_columns(_pl.col("round_num").cast(_pl.Int64)).join(fr, on=["match_id", "round_num"], how="left")
    if out["__freeze"].null_count():
        raise ValueError(f"{out['__freeze'].null_count()} rows without round metadata")
    out = out.with_columns(
        (_pl.col("start_tick") - _pl.col("__freeze")).alias("start_tick_norm"),
        (_pl.col("end_tick") - _pl.col("__freeze")).alias("end_tick_norm"),
    ).drop("__freeze")
    return out, True
