"""Compute video-to-trajectory time offsets for all CS2 round recordings.

Strategy
--------
The CS2 HUD shows a countdown timer at a fixed location on screen (format M:SS,
starting at 1:55 = trajectory time 0).  We read three digit slots:
  d0  minute     (624-632, 5-18)   – always "1" in CS2 rounds
  d1  tens-sec   (638-646, 5-18)
  d2  units-sec  (647-655, 5-18)

For each digit slot we keep pre-cropped grayscale templates stored in
  digit_templates/named/{d0,d1,d2}_<value>.png
Template matching (normalised cross-correlation) gives a robust read even
against the noisy/compressed background.

Alignment logic
---------------
Timer 1:55  ↔  trajectory game_sec = 0  (match start)
Timer 1:MM  ↔  game_sec = (115 - total_seconds)
             where total_seconds = (1*60 + MM)

The video starts recording some seconds *after* the match has begun, so the
first visible timer value is < 1:55.

We scan every frame (30 fps), read the timer, and look for the *first tick* –
the frame where the units digit first changes (from whatever the starting
value is to the next lower value).  That transition happens exactly at a
whole-second boundary in game time.

  game_time_of_transition = 115 - (first_timer_reading_after_transition)
    (because game_sec=0 ↔ 1:55=115 s, game_sec=1 ↔ 1:54=114 s, …)

  video_time_of_transition = transition_frame / video_fps

  offset = video_time_of_transition - game_time_of_transition

With this offset:  game_sec - offset = video_time

Output
------
For every round folder that has at least one video+parquet pair, writes/updates
  data/<videos|state_action>/match=.../round=<N>/metadata.json

The metadata dict maps player_id → {offset_sec, video_fps, start_timer,
transition_frame, transition_video_sec, transition_game_sec}.

A shared "round_offset" key (mean across players) is also stored for use in
multi-player video grid alignment.

Usage
-----
  python scripts/compute_offset.py            # process all rounds
  python scripts/compute_offset.py --verbose  # print per-frame debug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
STATE_ACTION_DIR = DATA_DIR / "state_action"
TEMPLATES_DIR = BASE_DIR / "digit_templates"

# ---------------------------------------------------------------------------
# Digit crop coordinates (in original 1280×720 frame)
# ---------------------------------------------------------------------------
DIGIT_BOXES = [
    (624, 5, 632, 18),   # d0 – minute
    (638, 5, 646, 18),   # d1 – tens of seconds
    (647, 5, 655, 18),   # d2 – units of seconds
]

VIDEO_FPS_NOMINAL = 30.0
TIMER_AT_ZERO = 115  # 1:55 in total seconds


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------
def _load_templates() -> dict[int, np.ndarray]:
    """Return {digit: gray_template_array} for digits 0-9 (shared font across all slots)."""
    templates: dict[int, np.ndarray] = {}
    for d in range(10):
        p = TEMPLATES_DIR / f"{d}.png"
        if not p.exists():
            raise FileNotFoundError(f"Template missing: {p}")
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        templates[d] = img.astype(np.float32)
    return templates


def _match_digit(crop_gray: np.ndarray, templates: dict[int, np.ndarray]) -> int:
    """Return the best-matching digit for a cropped gray region."""
    best_val = -np.inf
    best_digit = -1
    crop_f = crop_gray.astype(np.float32)
    for digit, tmpl in templates.items():
        if tmpl.shape != crop_f.shape:
            tmpl_r = cv2.resize(tmpl, (crop_f.shape[1], crop_f.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        else:
            tmpl_r = tmpl
        res = cv2.matchTemplate(crop_f, tmpl_r, cv2.TM_CCOEFF_NORMED)
        score = float(res[0, 0])
        if score > best_val:
            best_val = score
            best_digit = digit
    return best_digit


def read_timer(frame: np.ndarray,
               templates: dict[int, np.ndarray]) -> tuple[int, int, int]:
    """Return (minute, tens, units) digits read from frame."""
    digits = []
    for x1, y1, x2, y2 in DIGIT_BOXES:
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        digits.append(_match_digit(gray, templates))
    return tuple(digits)  # type: ignore[return-value]


def timer_to_game_sec(minute: int, tens: int, units: int) -> float:
    """Convert timer reading to game_sec (0 at 1:55)."""
    total = minute * 60 + tens * 10 + units
    return float(TIMER_AT_ZERO - total)


# ---------------------------------------------------------------------------
# Per-video offset computation
# ---------------------------------------------------------------------------
def compute_offset_for_video(video_path: Path,
                              templates: dict[int, np.ndarray],
                              verbose: bool = False) -> dict | None:
    """Scan video frames and return offset metadata dict, or None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path.name}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS_NOMINAL
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_units: int | None = None
    transition_frame: int | None = None
    timer_at_start: str | None = None
    timer_after_transition: str | None = None

    for fn in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        m, t, u = read_timer(frame, templates)
        timer_str = f"{m}:{t}{u}"

        if verbose:
            print(f"    frame {fn:4d}  timer={timer_str}")

        if timer_at_start is None:
            timer_at_start = timer_str

        if prev_units is not None and u != prev_units:
            # First digit change detected
            transition_frame = fn
            timer_after_transition = timer_str
            if verbose:
                print(f"  -> transition at frame {fn}: {prev_units} -> {u}  ({timer_str})")
            break

        prev_units = u

    cap.release()

    if transition_frame is None:
        print(f"  [WARN] No timer transition found in {video_path.name}")
        return None

    m2, t2, u2 = int(timer_after_transition[0]), int(timer_after_transition[2]), int(timer_after_transition[3])
    game_sec_at_transition = timer_to_game_sec(m2, t2, u2)
    video_sec_at_transition = transition_frame / fps

    # offset: add to game_sec to get video_time
    offset_sec = video_sec_at_transition - game_sec_at_transition

    print(f"  {video_path.name}: start={timer_at_start}  transition@f{transition_frame}"
          f"  game={game_sec_at_transition:.3f}s  video={video_sec_at_transition:.3f}s"
          f"  offset={offset_sec:+.4f}s")

    return {
        "offset_sec": round(offset_sec, 6),
        "video_fps": round(fps, 6),
        "start_timer": timer_at_start,
        "transition_timer": timer_after_transition,
        "transition_frame": transition_frame,
        "transition_video_sec": round(video_sec_at_transition, 6),
        "transition_game_sec": round(game_sec_at_transition, 6),
    }


# ---------------------------------------------------------------------------
# Round discovery and main loop
# ---------------------------------------------------------------------------
def iter_rounds():
    """Yield (match_id, round_num, round_video_dir, round_sa_dir) for all rounds."""
    if not VIDEOS_DIR.exists():
        return
    for match_dir in sorted(VIDEOS_DIR.iterdir()):
        if not match_dir.is_dir() or not match_dir.name.startswith("match="):
            continue
        match_id = match_dir.name.removeprefix("match=")
        for round_dir in sorted(match_dir.iterdir()):
            if not round_dir.is_dir() or not round_dir.name.startswith("round="):
                continue
            round_num = round_dir.name.removeprefix("round=")
            sa_dir = STATE_ACTION_DIR / f"match={match_id}" / f"round={round_num}"
            yield match_id, round_num, round_dir, sa_dir


def process_all(verbose: bool = False) -> None:
    print("Loading digit templates …")
    templates = _load_templates()
    print(f"Loaded templates for digits: {sorted(templates.keys())}")

    for match_id, round_num, vid_dir, sa_dir in iter_rounds():
        print(f"\n{'='*60}")
        print(f"Match {match_id}  round {round_num}")

        metadata: dict[str, object] = {}
        player_offsets: list[float] = []

        for video_file in sorted(vid_dir.glob("*.mp4")):
            player_id = video_file.stem
            pq = sa_dir / f"{player_id}.parquet"
            if not pq.exists():
                print(f"  [SKIP] No parquet for {player_id}")
                continue

            print(f"  Processing {player_id} …")
            result = compute_offset_for_video(video_file, templates, verbose=verbose)
            if result:
                metadata[player_id] = result
                player_offsets.append(result["offset_sec"])

        if not metadata:
            print("  [SKIP] No valid players found, skipping metadata write.")
            continue

        # Round-level offset: mean across players (for grid alignment)
        metadata["round_offset_sec"] = round(float(np.mean(player_offsets)), 6)

        # Write to video round dir
        meta_path = vid_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        print(f"\n  Wrote {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute video-trajectory offsets.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-frame timer readings.")
    args = parser.parse_args()
    process_all(verbose=args.verbose)
