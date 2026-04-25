"""Check time_offset.json coverage against videos, trajectories, and contrastive labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, NamedTuple

import cv2
from dotenv import load_dotenv


class ClipKey(NamedTuple):
    match_id: str
    player_id: str
    round_name: str

    def short(self) -> str:
        return f"{self.match_id}/{self.player_id}/{self.round_name}"


def _round_name(value: str | int) -> str:
    text = str(value)
    return text if text.startswith("round_") else f"round_{int(float(text))}"


def _resolve_data_root(cli_data_root: str | None) -> Path:
    load_dotenv()
    data_root = Path(cli_data_root or os.getenv("DATA_BASE_PATH", "data"))
    if data_root.is_absolute():
        return data_root
    return Path(__file__).resolve().parents[3] / data_root


def _iter_file_tree(root: Path, suffix: str) -> Iterable[ClipKey]:
    if not root.exists():
        return
    for path in root.glob(f"*/*/*{suffix}"):
        if not path.is_file():
            continue
        yield ClipKey(
            match_id=path.parent.parent.name,
            player_id=path.parent.name,
            round_name=path.stem,
        )


def _iter_offset_keys(offsets: dict) -> Iterable[ClipKey]:
    for match_id, players in offsets.items():
        if not isinstance(players, dict):
            continue
        for player_id, rounds in players.items():
            if not isinstance(rounds, dict):
                continue
            for round_name in rounds:
                yield ClipKey(str(match_id), str(player_id), str(round_name))


@lru_cache(maxsize=8192)
def _video_duration_seconds(video_path: str) -> float | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if not fps or fps <= 0 or not frame_count or frame_count <= 0:
        return None
    return float(frame_count / fps)


def _load_label_refs(label_path: Path) -> tuple[list[tuple[ClipKey, dict]], Counter]:
    refs: list[tuple[ClipKey, dict]] = []
    partitions: Counter = Counter()

    if not label_path.exists():
        return refs, partitions

    with open(label_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            partition = row.get("partition", "")
            partitions[partition] += 1
            num_alive = int(float(row["num_alive_teammates"]))
            for i in range(num_alive):
                player_id = row.get(f"teammate_{i}_id", "").strip()
                if not player_id:
                    continue
                key = ClipKey(
                    match_id=row["match_id"],
                    player_id=player_id,
                    round_name=_round_name(row["round_num"]),
                )
                refs.append((key, row))

    return refs, partitions


def _print_examples(title: str, items: list[str], limit: int) -> None:
    print(f"  {title}: {len(items)}")
    for item in items[:limit]:
        print(f"    - {item}")
    if len(items) > limit:
        print(f"    ... {len(items) - limit} more")


def check_map(
    data_root: Path,
    map_name: str,
    labels_filename: str,
    video_folder: str,
    trajectory_folder: str,
    limit: int,
) -> int:
    map_root = data_root / map_name
    offset_path = map_root / "time_offset.json"
    label_path = map_root / "labels" / labels_filename
    video_root = map_root / video_folder
    trajectory_root = map_root / trajectory_folder

    print(f"\n=== {map_name} ===")
    print(f"offsets: {offset_path}")
    print(f"labels : {label_path}")
    print(f"videos : {video_root}")
    print(f"traj   : {trajectory_root}")

    if not offset_path.exists():
        print("  ERROR: missing time_offset.json")
        return 1

    offsets = json.loads(offset_path.read_text(encoding="utf-8"))
    offset_keys = set(_iter_offset_keys(offsets))
    video_keys = set(_iter_file_tree(video_root, ".mp4"))
    trajectory_keys = set(_iter_file_tree(trajectory_root, ".csv"))
    label_refs, label_partitions = _load_label_refs(label_path)
    label_keys = {key for key, _ in label_refs}

    print(f"  offset entries: {len(offset_keys)}")
    print(f"  video files   : {len(video_keys)}")
    print(f"  trajectory csv: {len(trajectory_keys)}")
    print(f"  label rows    : {sum(label_partitions.values())} {dict(label_partitions)}")
    print(f"  label refs    : {len(label_refs)}")

    problems = 0
    video_missing_offset = sorted(key.short() for key in video_keys - offset_keys)
    offset_missing_video = sorted(key.short() for key in offset_keys - video_keys)
    label_missing_offset = sorted(key.short() for key in label_keys - offset_keys)
    label_missing_video = sorted(key.short() for key in label_keys - video_keys)
    label_missing_trajectory = sorted(key.short() for key in label_keys - trajectory_keys)

    for title, items in [
        ("video files missing offset entries", video_missing_offset),
        ("offset entries missing video files", offset_missing_video),
        ("label refs missing offset entries", label_missing_offset),
        ("label refs missing video files", label_missing_video),
        ("label refs missing trajectory files", label_missing_trajectory),
    ]:
        if items:
            problems += len(items)
        _print_examples(title, items, limit)

    incomplete_examples: list[str] = []
    incomplete_by_partition: Counter = Counter()
    for key, row in label_refs:
        try:
            offset_sec = float(offsets[key.match_id][key.player_id][key.round_name]["offset_sec"])
        except KeyError:
            continue

        video_path = video_root / key.match_id / key.player_id / f"{key.round_name}.mp4"
        if not video_path.exists():
            continue

        duration = _video_duration_seconds(str(video_path))
        if duration is None:
            incomplete_by_partition[row.get("partition", "")] += 1
            if len(incomplete_examples) < limit:
                incomplete_examples.append(f"{key.short()} idx={row.get('idx')} unreadable_video")
            continue

        start = float(row["start_seconds"]) + offset_sec
        end = float(row["end_seconds"]) + offset_sec
        if start < 0 or end > duration:
            incomplete_by_partition[row.get("partition", "")] += 1
            if len(incomplete_examples) < limit:
                incomplete_examples.append(
                    f"{key.short()} idx={row.get('idx')} partition={row.get('partition')} "
                    f"video=[{start:.3f},{end:.3f}] duration={duration:.3f}"
                )

    incomplete_count = sum(incomplete_by_partition.values())
    problems += incomplete_count
    print(f"  label refs without full aligned clip: {incomplete_count} {dict(incomplete_by_partition)}")
    for item in incomplete_examples:
        print(f"    - {item}")
    if incomplete_count > len(incomplete_examples):
        print(f"    ... {incomplete_count - len(incomplete_examples)} more")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=None, help="Defaults to DATA_BASE_PATH or ./data")
    parser.add_argument("--maps", nargs="+", default=["dust2", "inferno", "mirage"])
    parser.add_argument("--labels_filename", default="contrastive.csv")
    parser.add_argument("--video_folder", default="video_306x306_4fps")
    parser.add_argument("--trajectory_folder", default="trajectory")
    parser.add_argument("--limit", type=int, default=20, help="Examples to print per issue type")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any issue is found")
    args = parser.parse_args()

    data_root = _resolve_data_root(args.data_root)
    total_problems = 0
    for map_name in args.maps:
        total_problems += check_map(
            data_root=data_root,
            map_name=map_name,
            labels_filename=args.labels_filename,
            video_folder=args.video_folder,
            trajectory_folder=args.trajectory_folder,
            limit=args.limit,
        )

    print(f"\nTotal problems: {total_problems}")
    if args.strict and total_problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
