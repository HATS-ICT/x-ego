import os
import sys
import json
import glob
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Config
BASE_DIR = Path(r"C:\Users\wangy\projects\x-ego")
sys.path.append(str(BASE_DIR))

try:
    from src.scripts.task_creator.task_definitions import (
        MOVEMENT_DIRECTIONS,
        ROUND_OUTCOMES,
        get_place_names_for_map,
    )
except ImportError as e:
    print(f"Warning: Could not import task definitions: {e}")
    MOVEMENT_DIRECTIONS = []
    ROUND_OUTCOMES = []
    def get_place_names_for_map(_map_name):
        return []

TEST_DATA_DIR = BASE_DIR / "temp_test_video_label" / "data"
VIDEO_DIR = Path(r"E:\files\data\cs101\recording\video")

# Ensure test data dir exists
os.makedirs(TEST_DATA_DIR, exist_ok=True)

DATA_ROOT = BASE_DIR / "data"
MAPS = ["dust2", "inferno", "mirage"]


def is_snapshot_task(task_name: str) -> bool:
    """Return True for tasks whose labels correspond to a single prediction tick."""
    task_lower = task_name.lower()
    return "location" in task_lower

# Load all offsets to memory
time_offsets = {}
map_by_match = {}
metadata_cache = {}
kill_events_cache = {}

for m in MAPS:
    offset_file = DATA_ROOT / m / "time_offset.json"
    if offset_file.exists():
        try:
            with open(offset_file, "r") as f:
                data = json.load(f)
                time_offsets[m] = data
                for match_id in data.keys():
                    map_by_match[match_id] = m
        except Exception as e:
            print(f"Error loading {offset_file}: {e}")


def get_round_freeze_end_tick(map_name: str, match_id: str, round_num: int):
    cache_key = (map_name, match_id)
    if cache_key not in metadata_cache:
        metadata_file = DATA_ROOT / map_name / "metadata" / f"{match_id}.json"
        if not metadata_file.exists():
            metadata_cache[cache_key] = None
        else:
            try:
                with open(metadata_file, "r") as f:
                    metadata_cache[cache_key] = json.load(f)
            except Exception as e:
                print(f"Error loading {metadata_file}: {e}")
                metadata_cache[cache_key] = None

    metadata = metadata_cache.get(cache_key)
    if not metadata:
        return None

    round_info = next((r for r in metadata.get("rounds", []) if r.get("round_number") == round_num), None)
    if not round_info:
        return None
    return round_info.get("freeze_end_tick")


def tick_to_game_seconds(raw_tick, freeze_end_tick):
    if pd.isnull(raw_tick):
        return None
    tick = float(raw_tick)
    if freeze_end_tick is not None and tick >= freeze_end_tick:
        tick -= freeze_end_tick
    return tick / 64.0


def get_first_kill_event_tick(
    map_name: str,
    match_id: str,
    round_num: int,
    start_tick,
    end_tick,
    steamid_col: str = None,
    steamid: str = None,
):
    cache_key = (map_name, match_id)
    if cache_key not in kill_events_cache:
        kills_file = DATA_ROOT / map_name / "event" / match_id / "kills.csv"
        if not kills_file.exists():
            kill_events_cache[cache_key] = pd.DataFrame()
        else:
            try:
                kill_events_cache[cache_key] = pd.read_csv(kills_file)
            except Exception as e:
                print(f"Error loading {kills_file}: {e}")
                kill_events_cache[cache_key] = pd.DataFrame()

    kills_df = kill_events_cache.get(cache_key, pd.DataFrame())
    if kills_df.empty or pd.isnull(start_tick) or pd.isnull(end_tick):
        return None

    tick_col = "tick_norm" if "tick_norm" in kills_df.columns else "tick"
    round_kills = kills_df[kills_df["round_num"] == round_num]
    if steamid_col and steamid_col in round_kills.columns and steamid is not None:
        round_kills = round_kills[round_kills[steamid_col].astype(str) == str(steamid)]
    window_kills = round_kills[(round_kills[tick_col] >= start_tick) & (round_kills[tick_col] <= end_tick)]
    if window_kills.empty:
        return None
    first_kill = window_kills.sort_values(tick_col).iloc[0]
    raw_tick = first_kill.get("tick", first_kill[tick_col])
    norm_tick = first_kill.get("tick_norm", first_kill[tick_col])
    return float(raw_tick), float(norm_tick)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/videos")
def list_videos():
    # Scan TEST_DATA_DIR for mp4 files
    videos = []
    for root, dirs, files in os.walk(TEST_DATA_DIR):
        for file in files:
            if file.endswith(".mp4"):
                path = Path(root) / file
                rel_path = path.relative_to(TEST_DATA_DIR)
                videos.append(str(rel_path).replace("\\", "/"))
    return jsonify(videos)

@app.route("/api/video_info")
def get_video_info():
    rel_path = request.args.get("video")
    if not rel_path:
        return jsonify({"error": "Missing video"}), 400
    
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        match_id = parts[-3]
        map_name = map_by_match.get(match_id)
        if map_name:
            return jsonify({"map": map_name})
    
    # Fallback: maybe just filename has match_id? (unlikely but safe)
    for m, matches in time_offsets.items():
        for match_id in matches.keys():
            if match_id in rel_path:
                return jsonify({"map": m})
                
    return jsonify({"error": "Map not found"}), 404

@app.route("/api/tasks")
def list_tasks():
    map_name = request.args.get("map")
    if not map_name or map_name not in MAPS:
        return jsonify([])
    task_dir = DATA_ROOT / map_name / "labels" / "all_tasks"
    tasks = []
    if task_dir.exists():
        for f in task_dir.glob("*.csv"):
            tasks.append(f.stem)
    return jsonify(sorted(tasks))

@app.route("/video/<path:rel_path>")
def serve_video(rel_path):
    video_path = TEST_DATA_DIR / rel_path
    if not video_path.exists():
        video_path = VIDEO_DIR / rel_path
    return send_file(video_path)

@app.route("/api/label_data")
def get_label_data():
    rel_path = request.args.get("video")
    task = request.args.get("task")
    
    if not rel_path or not task:
        return jsonify({"error": "Missing video or task"}), 400
        
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        match_id = parts[-3]
        player_id = parts[-2]
        round_name = parts[-1].replace(".mp4", "")
        try:
            round_num = int(round_name.replace("round_", ""))
        except:
            return jsonify({"error": "Could not parse round number"}), 400
    else:
        return jsonify({"error": "Invalid video path structure. Must be match_id/player_id/round_X.mp4"}), 400
        
    map_name = map_by_match.get(match_id)
    if not map_name:
        # Fallback search
        for m, matches in time_offsets.items():
            if match_id in matches:
                map_name = m
                break
        if not map_name:
            return jsonify({"error": f"Map not found for match {match_id}"}), 404
        
    offset_sec = time_offsets.get(map_name, {}).get(match_id, {}).get(player_id, {}).get(round_name, {}).get("offset_sec", 0.0)
    
    csv_path = DATA_ROOT / map_name / "labels" / "all_tasks" / f"{task}.csv"
    if not csv_path.exists():
        return jsonify({"error": f"Task CSV not found: {csv_path}"}), 404
        
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
        # Filter for this exact video
        df = df[(df["match_id"] == match_id) & (df["round_num"] == round_num) & (df["pov_steamid"] == int(player_id))]
        
        records = []
        std_cols = {"match_id", "round_num", "pov_steamid", "start_tick", "end_tick", "partition", "idx"}
        
        snapshot_task = is_snapshot_task(task)

        place_names = get_place_names_for_map(map_name)
        freeze_end_tick = get_round_freeze_end_tick(map_name, match_id, round_num)

        for _, row in df.iterrows():
            start_tick_value = row.get("start_tick_norm")
            end_tick_value = row.get("end_tick_norm")
            prediction_tick_value = row.get("prediction_tick_norm")

            if pd.isnull(start_tick_value):
                start_tick_value = row.get("start_tick")
            if pd.isnull(end_tick_value):
                end_tick_value = row.get("end_tick")

            game_start = tick_to_game_seconds(start_tick_value, None)
            game_end = tick_to_game_seconds(end_tick_value, None)
            prediction_tick = row.get("prediction_tick")
            prediction_tick_norm = row.get("prediction_tick_norm")
            horizon_sec = row.get("horizon_sec")
            label_value = row.get("label")
            first_kill_event_ticks = None
            if task.lower().startswith("global_anykill") and label_value == 1:
                first_kill_event_ticks = get_first_kill_event_tick(
                    map_name,
                    match_id,
                    round_num,
                    row.get("end_tick_norm") if pd.notnull(row.get("end_tick_norm")) else row.get("end_tick"),
                    (
                        row.get("end_tick_norm") if pd.notnull(row.get("end_tick_norm")) else row.get("end_tick")
                    ) + (horizon_sec * 64 if pd.notnull(horizon_sec) else 0),
                )
            elif task.lower().startswith("self_kill") and row.get("label_pov_kills") == 1:
                first_kill_event_ticks = get_first_kill_event_tick(
                    map_name,
                    match_id,
                    round_num,
                    row.get("end_tick_norm") if pd.notnull(row.get("end_tick_norm")) else row.get("end_tick"),
                    (
                        row.get("end_tick_norm") if pd.notnull(row.get("end_tick_norm")) else row.get("end_tick")
                    ) + (horizon_sec * 64 if pd.notnull(horizon_sec) else 0),
                    "attacker_steamid",
                    player_id,
                )
            elif task.lower().startswith("self_death") and label_value == 1:
                first_kill_event_ticks = get_first_kill_event_tick(
                    map_name,
                    match_id,
                    round_num,
                    row.get("end_tick_norm") if pd.notnull(row.get("end_tick_norm")) else row.get("end_tick"),
                    (
                        row.get("end_tick_norm") if pd.notnull(row.get("end_tick_norm")) else row.get("end_tick")
                    ) + (horizon_sec * 64 if pd.notnull(horizon_sec) else 0),
                    "victim_steamid",
                    player_id,
                )

            if first_kill_event_ticks is not None:
                prediction_tick, prediction_tick_value = first_kill_event_ticks
                prediction_tick_norm = prediction_tick_value

            has_prediction_tick = pd.notnull(prediction_tick) and not (
                (task.lower().startswith("global_anykill") and label_value == 0)
                or (task.lower().startswith("self_kill") and row.get("label_pov_kills") == 0)
                or (task.lower().startswith("self_death") and label_value == 0)
            )
            if pd.notnull(prediction_tick_value):
                game_prediction = tick_to_game_seconds(prediction_tick_value, None)
            elif has_prediction_tick and pd.notnull(row.get("start_tick")):
                game_prediction = tick_to_game_seconds(prediction_tick, freeze_end_tick)
            else:
                game_prediction = (game_start + game_end) / 2.0
            
            video_start = game_start + offset_sec
            video_end = game_end + offset_sec
            video_prediction = game_prediction + offset_sec
            
            raw_labels = {k: v for k, v in row.items() if k not in std_cols}
            interpreted_labels = {}
            task_lower = task.lower()
            
            # Handle locations (multi-label)
            if "location" in task_lower and "label" not in raw_labels:
                locs = []
                for i in range(len(place_names)):
                    if raw_labels.get(f"label_{i}", 0) == 1:
                        locs.append(place_names[i])
                interpreted_labels["Locations"] = locs
            
            for k, v in raw_labels.items():
                if k == "label":
                    if "location" in task_lower:
                        idx = int(v)
                        interpreted_labels["Location"] = place_names[idx] if 0 <= idx < len(place_names) else f"Unknown ({idx})"
                    elif task_lower.startswith("self_death"):
                        interpreted_labels["POV Dies"] = "Yes" if v == 1 else "No"
                    elif task_lower.startswith("global_anykill"):
                        interpreted_labels["Any Kill"] = "Yes" if v == 1 else "No"
                    elif "movementdir" in task_lower or "movement_dir" in task_lower:
                        idx = int(v)
                        interpreted_labels["Direction"] = MOVEMENT_DIRECTIONS[idx] if 0 <= idx < len(MOVEMENT_DIRECTIONS) else f"Unknown ({idx})"
                    elif "alivecount" in task_lower:
                        interpreted_labels["Alive Count"] = int(v)
                    else:
                        interpreted_labels[k] = v
                elif k == "label_pov_kills":
                    interpreted_labels["POV Kills"] = "Yes" if v == 1 else "No"
                elif k == "label_outcome":
                    interpreted_labels["Outcome"] = "Exploded" if v == 1 else "Defused"
                elif k == "label_round_winner":
                    interpreted_labels["Winner"] = "T" if v == 1 else "CT"
                elif k == "label_outcome_reason":
                    idx = int(v)
                    interpreted_labels["Reason"] = ROUND_OUTCOMES[idx] if 0 <= idx < len(ROUND_OUTCOMES) else f"Unknown ({idx})"
                elif k == "label_bomb_site":
                    interpreted_labels["Site"] = "B" if v == 1 else "A"
                elif k == "label_bomb_planted":
                    interpreted_labels["Planted"] = "Yes" if v == 1 else "No"
                elif not k.startswith("label_"):
                    interpreted_labels[k] = v
            
            records.append({
                "video_start": video_start,
                "video_end": video_end,
                "video_prediction": video_prediction,
                "game_start": game_start,
                "game_end": game_end,
                "game_prediction": game_prediction,
                "is_snapshot": snapshot_task,
                "has_prediction_tick": bool(has_prediction_tick),
                "labels": interpreted_labels,
                "idx": int(row.get("idx", 0)) if "idx" in row and pd.notnull(row["idx"]) else "N/A",
                "partition": row.get("partition", "N/A"),
                "start_tick": int(row.get("start_tick", 0)) if "start_tick" in row and pd.notnull(row["start_tick"]) else "N/A",
                "end_tick": int(row.get("end_tick", 0)) if "end_tick" in row and pd.notnull(row["end_tick"]) else "N/A",
                "prediction_tick": int(prediction_tick) if pd.notnull(prediction_tick) else "N/A",
                "start_tick_norm": int(start_tick_value) if pd.notnull(start_tick_value) else "N/A",
                "end_tick_norm": int(end_tick_value) if pd.notnull(end_tick_value) else "N/A",
                "prediction_tick_norm": int(prediction_tick_norm) if pd.notnull(prediction_tick_norm) else "N/A",
                "match_id": str(row.get("match_id", "N/A")),
                "round_num": int(row.get("round_num", 0)) if "round_num" in row and pd.notnull(row["round_num"]) else "N/A",
                "pov_steamid": str(int(row.get("pov_steamid", 0))) if "pov_steamid" in row and pd.notnull(row["pov_steamid"]) else "N/A"
            })
            
        records.sort(key=lambda x: x["video_start"])
        
        return jsonify({
            "map": map_name,
            "offset_sec": offset_sec,
            "records": records
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
