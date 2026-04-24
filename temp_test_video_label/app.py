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

        for _, row in df.iterrows():
            game_start = row["start_tick_norm"] / 64.0
            game_end = row["end_tick_norm"] / 64.0
            prediction_tick_norm = row.get("prediction_tick_norm")
            game_prediction = (
                prediction_tick_norm / 64.0
                if pd.notnull(prediction_tick_norm)
                else (game_start + game_end) / 2.0
            )
            
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
                    elif "movementdir" in task_lower or "movement_dir" in task_lower:
                        idx = int(v)
                        interpreted_labels["Direction"] = MOVEMENT_DIRECTIONS[idx] if 0 <= idx < len(MOVEMENT_DIRECTIONS) else f"Unknown ({idx})"
                    elif "alivecount" in task_lower:
                        interpreted_labels["Alive Count"] = int(v)
                    else:
                        interpreted_labels[k] = v
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
                "labels": interpreted_labels,
                "idx": int(row.get("idx", 0)) if "idx" in row and pd.notnull(row["idx"]) else "N/A",
                "partition": row.get("partition", "N/A"),
                "start_tick": int(row.get("start_tick", 0)) if "start_tick" in row and pd.notnull(row["start_tick"]) else "N/A",
                "end_tick": int(row.get("end_tick", 0)) if "end_tick" in row and pd.notnull(row["end_tick"]) else "N/A",
                "prediction_tick": int(row.get("prediction_tick", 0)) if "prediction_tick" in row and pd.notnull(row["prediction_tick"]) else "N/A",
                "start_tick_norm": int(row.get("start_tick_norm", 0)) if "start_tick_norm" in row and pd.notnull(row["start_tick_norm"]) else "N/A",
                "end_tick_norm": int(row.get("end_tick_norm", 0)) if "end_tick_norm" in row and pd.notnull(row["end_tick_norm"]) else "N/A",
                "prediction_tick_norm": int(row.get("prediction_tick_norm", 0)) if "prediction_tick_norm" in row and pd.notnull(row["prediction_tick_norm"]) else "N/A",
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
