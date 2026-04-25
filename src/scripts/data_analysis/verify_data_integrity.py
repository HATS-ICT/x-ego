"""Verify that training/validation/test data references existing files.

This is a lightweight preflight check for the contrastive pipeline. It verifies:
- each map label CSV exists
- each map labels folder exists
- each map time_offset.json exists
- every referenced teammate video exists
- every referenced teammate has an offset entry
- every referenced teammate trajectory round CSV exists
- every referenced match has a metadata JSON and demo file
- every referenced match has an event folder with expected CSVs

Example:
    uv run python src/scripts/data_analysis/verify_data_integrity.py --strict
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MAPS = ("dust2", "inferno", "mirage")
DEFAULT_PARTITIONS = ("train", "val", "test")
DEFAULT_EVENT_FILES = ("bomb.csv", "damages.csv", "kills.csv", "rounds.csv", "shots.csv")


def resolve_data_root(cli_data_root: str | None) -> Path:
    load_dotenv()
    data_root = Path(cli_data_root or os.getenv("DATA_BASE_PATH", "data"))
    if data_root.is_absolute():
        return data_root
    return Path(__file__).resolve().parents[3] / data_root


def round_key(round_num: str) -> str:
    text = str(round_num)
    if text.startswith("round_"):
        return text
    return f"round_{int(float(text))}"


def has_offset(offsets: dict, match_id: str, player_id: str, round_name: str) -> bool:
    return round_name in offsets.get(str(match_id), {}).get(str(player_id), {})


def print_examples(title: str, examples: list[str], limit: int) -> None:
    print(f"  {title}: {len(examples)}")
    for example in examples[:limit]:
        print(f"    - {example}")
    if len(examples) > limit:
        print(f"    ... {len(examples) - limit} more")


def check_contrastive_map(
    data_root: Path,
    map_name: str,
    labels_filename: str,
    video_folder: str,
    trajectory_folder: str,
    demo_folder: str,
    metadata_folder: str,
    event_folder: str,
    event_files: tuple[str, ...],
    partitions: set[str],
    limit: int,
) -> int:
    map_root = data_root / map_name
    labels_root = map_root / "labels"
    label_path = labels_root / labels_filename
    offset_path = map_root / "time_offset.json"
    video_root = map_root / video_folder
    trajectory_root = map_root / trajectory_folder
    demo_root = map_root / demo_folder
    metadata_root = map_root / metadata_folder
    event_root = map_root / event_folder

    print(f"\n=== {map_name} ===")
    print(f"labels folder: {labels_root}")
    print(f"labels: {label_path}")
    print(f"offset: {offset_path}")
    print(f"videos: {video_root}")
    print(f"trajectory: {trajectory_root}")
    print(f"demo: {demo_root}")
    print(f"metadata: {metadata_root}")
    print(f"event: {event_root}")

    problems = 0
    if not labels_root.exists():
        print("  ERROR: labels folder is missing")
        return 1
    if not label_path.exists():
        print("  ERROR: label CSV is missing")
        return 1
    if not offset_path.exists():
        print("  ERROR: time_offset.json is missing")
        return 1
    if not video_root.exists():
        print("  ERROR: video folder is missing")
        return 1
    if not trajectory_root.exists():
        print("  ERROR: trajectory folder is missing")
        return 1
    if not demo_root.exists():
        print("  ERROR: demo folder is missing")
        return 1
    if not metadata_root.exists():
        print("  ERROR: metadata folder is missing")
        return 1
    if not event_root.exists():
        print("  ERROR: event folder is missing")
        return 1

    offsets = json.loads(offset_path.read_text(encoding="utf-8"))

    row_count = 0
    checked_refs = 0
    partition_counts: Counter[str] = Counter()
    checked_matches: set[str] = set()
    missing_videos: list[str] = []
    missing_offsets: list[str] = []
    missing_trajectories: list[str] = []
    missing_demos: list[str] = []
    missing_metadata: list[str] = []
    missing_events: list[str] = []
    bad_rows: list[str] = []

    with label_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {
            "partition",
            "match_id",
            "round_num",
            "num_alive_teammates",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            print(f"  ERROR: missing required column(s): {sorted(missing_columns)}")
            return 1

        for csv_line, row in enumerate(reader, start=2):
            partition = row.get("partition", "")
            if partition not in partitions:
                continue

            row_count += 1
            partition_counts[partition] += 1
            match_id = row["match_id"]
            round_name = round_key(row["round_num"])
            checked_matches.add(str(match_id))

            try:
                num_alive = int(float(row["num_alive_teammates"]))
            except ValueError:
                bad_rows.append(f"line={csv_line} invalid num_alive_teammates={row.get('num_alive_teammates')!r}")
                problems += 1
                continue

            for i in range(num_alive):
                player_id = row.get(f"teammate_{i}_id", "").strip()
                if not player_id:
                    bad_rows.append(f"line={csv_line} missing teammate_{i}_id")
                    problems += 1
                    continue

                checked_refs += 1
                video_path = video_root / str(match_id) / str(player_id) / f"{round_name}.mp4"
                if not video_path.exists():
                    missing_videos.append(f"line={csv_line} {match_id}/{player_id}/{round_name}")
                    problems += 1

                trajectory_path = trajectory_root / str(match_id) / str(player_id) / f"{round_name}.csv"
                if not trajectory_path.exists():
                    missing_trajectories.append(f"line={csv_line} {match_id}/{player_id}/{round_name}")
                    problems += 1

                if not has_offset(offsets, match_id, player_id, round_name):
                    missing_offsets.append(f"line={csv_line} {match_id}/{player_id}/{round_name}")
                    problems += 1

    for match_id in sorted(checked_matches):
        demo_path = demo_root / f"{match_id}.dem"
        if not demo_path.exists():
            missing_demos.append(match_id)
            problems += 1

        metadata_path = metadata_root / f"{match_id}.json"
        if not metadata_path.exists():
            missing_metadata.append(match_id)
            problems += 1

        match_event_root = event_root / match_id
        if not match_event_root.exists():
            missing_events.append(f"{match_id}/")
            problems += 1
        else:
            for event_file in event_files:
                event_path = match_event_root / event_file
                if not event_path.exists():
                    missing_events.append(f"{match_id}/{event_file}")
                    problems += 1

    print(f"  rows checked: {row_count:,} {dict(partition_counts)}")
    print(f"  matches    : {len(checked_matches):,}")
    print(f"  video refs : {checked_refs:,}")
    print_examples("bad rows", bad_rows, limit)
    print_examples("missing videos", missing_videos, limit)
    print_examples("missing offsets", missing_offsets, limit)
    print_examples("missing trajectories", missing_trajectories, limit)
    print_examples("missing demos", missing_demos, limit)
    print_examples("missing metadata", missing_metadata, limit)
    print_examples("missing events", missing_events, limit)

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=None, help="Defaults to DATA_BASE_PATH or ./data")
    parser.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    parser.add_argument("--partitions", nargs="+", default=list(DEFAULT_PARTITIONS))
    parser.add_argument("--labels_filename", default="contrastive.csv")
    parser.add_argument("--video_folder", default="video_306x306_4fps")
    parser.add_argument("--trajectory_folder", default="trajectory")
    parser.add_argument("--demo_folder", default="demo")
    parser.add_argument("--metadata_folder", default="metadata")
    parser.add_argument("--event_folder", default="event")
    parser.add_argument("--event_files", nargs="+", default=list(DEFAULT_EVENT_FILES))
    parser.add_argument("--limit", type=int, default=20, help="Examples to print per issue type")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any problem is found")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    partitions = set(args.partitions)

    total_problems = 0
    for map_name in args.maps:
        total_problems += check_contrastive_map(
            data_root=data_root,
            map_name=map_name,
            labels_filename=args.labels_filename,
            video_folder=args.video_folder,
            trajectory_folder=args.trajectory_folder,
            demo_folder=args.demo_folder,
            metadata_folder=args.metadata_folder,
            event_folder=args.event_folder,
            event_files=tuple(args.event_files),
            partitions=partitions,
            limit=args.limit,
        )

    print(f"\nTotal problems: {total_problems}")
    if args.strict and total_problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
