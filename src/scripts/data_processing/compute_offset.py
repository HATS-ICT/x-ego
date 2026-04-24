"""Compute video-to-trajectory time offsets for all CS2 round recordings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEMPLATES_DIR = Path(r"C:\Users\wangy\projects\x-ego\src\digit_templates")

VIDEO_DIR = Path(r"E:\files\data\cs101\recording\video")
OUT_DATA_DIR = Path(r"c:\Users\wangy\projects\x-ego\data")

MAP_LIST_FILES = {
    'dust2': r'c:\Users\wangy\projects\x-ego\data\dust2\video_list.txt',
    'inferno': r'c:\Users\wangy\projects\x-ego\data\inferno\video_list.txt',
    'mirage': r'c:\Users\wangy\projects\x-ego\data\mirage\video_list.txt'
}

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
    templates: dict[int, np.ndarray] = {}
    for d in range(10):
        p = TEMPLATES_DIR / f"{d}.png"
        if not p.exists():
            raise FileNotFoundError(f"Template missing: {p}")
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        templates[d] = img.astype(np.float32)
    return templates


def _match_digit(crop_gray: np.ndarray, templates: dict[int, np.ndarray]) -> int:
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
    digits = []
    for x1, y1, x2, y2 in DIGIT_BOXES:
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        digits.append(_match_digit(gray, templates))
    return tuple(digits)  # type: ignore[return-value]


def timer_to_game_sec(minute: int, tens: int, units: int) -> float:
    total = minute * 60 + tens * 10 + units
    return float(TIMER_AT_ZERO - total)


# ---------------------------------------------------------------------------
# Per-video offset computation
# ---------------------------------------------------------------------------
def compute_offset_for_video(video_path: Path,
                              templates: dict[int, np.ndarray],
                              verbose: bool = False) -> dict | None:
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

    offset_sec = video_sec_at_transition - game_sec_at_transition

    print(f"    start={timer_at_start}  transition@f{transition_frame}"
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


def load_map_match_mapping() -> dict[str, str]:
    mapping = {}
    for map_name, list_file in MAP_LIST_FILES.items():
        if not os.path.exists(list_file):
            continue
        with open(list_file, 'rb') as f:
            raw_data = f.read()
        if raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'):
            encoding = 'utf-16'
        else:
            encoding = 'utf-8'
        with open(list_file, 'r', encoding=encoding) as f:
            for line in f:
                vid = line.strip()
                if vid:
                    mapping[vid] = map_name
    return mapping


def process_all(verbose: bool = False) -> None:
    print("Loading digit templates …")
    templates = _load_templates()
    print(f"Loaded templates for digits: {sorted(templates.keys())}")

    mapping = load_map_match_mapping()
    
    results = {'dust2': {}, 'inferno': {}, 'mirage': {}}
    
    if not VIDEO_DIR.exists():
        print(f"Video directory not found: {VIDEO_DIR}")
        return

    for match_dir in sorted(VIDEO_DIR.iterdir()):
        if not match_dir.is_dir():
            continue
            
        match_id = match_dir.name
        map_name = mapping.get(match_id, 'unknown')
        
        if map_name not in results:
            results[map_name] = {}
            
        if match_id not in results[map_name]:
            results[map_name][match_id] = {}
            
        print(f"\n{'='*60}")
        print(f"Match {match_id} (Map: {map_name})")
        
        for player_dir in sorted(match_dir.iterdir()):
            if not player_dir.is_dir():
                continue
            player_id = player_dir.name
            
            if player_id not in results[map_name][match_id]:
                results[map_name][match_id][player_id] = {}
                
            for video_file in sorted(player_dir.glob("*.mp4")):
                round_name = video_file.stem
                print(f"  Processing {player_id} / {round_name} …")
                
                result = compute_offset_for_video(video_file, templates, verbose=verbose)
                if result:
                    results[map_name][match_id][player_id][round_name] = result

    print("\nSaving results...")
    for map_name, data in results.items():
        if not data:
            continue
            
        out_file = OUT_DATA_DIR / map_name / "time_offset.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(data, indent=2))
        print(f"Saved {len(data)} matches for map '{map_name}' to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute video-trajectory offsets.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-frame timer readings.")
    args = parser.parse_args()
    process_all(verbose=args.verbose)
