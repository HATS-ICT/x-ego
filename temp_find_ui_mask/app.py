import base64
import json
import mimetypes
import time
from pathlib import Path
from uuid import uuid4

from flask import Response, Flask, jsonify, render_template, request, send_file


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
VIDEO_DIR = DATA_DIR / "videos"
MASKS_PATH = DATA_DIR / "masks.json"

RAW_WIDTH = 1280
RAW_HEIGHT = 720
RESIZED_WIDTH = 306
RESIZED_HEIGHT = 306
VIDEO_WIDTH = 544
VIDEO_HEIGHT = 306

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv"}

app = Flask(__name__)


def ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not MASKS_PATH.exists():
        save_state({"images": [], "boxes_by_image": {}})


def load_state() -> dict:
    ensure_dirs()
    try:
        with MASKS_PATH.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        state = {"images": [], "boxes_by_image": {}}
    state.setdefault("images", [])
    state.setdefault("boxes_by_image", {})
    return state


def save_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with MASKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def normalized_box(box: dict) -> dict:
    x1 = int(round(float(box["x"])))
    y1 = int(round(float(box["y"])))
    x2 = int(round(float(box["x"]) + float(box["w"])))
    y2 = int(round(float(box["y"]) + float(box["h"])))
    left = max(0, min(RAW_WIDTH, min(x1, x2)))
    top = max(0, min(RAW_HEIGHT, min(y1, y2)))
    right = max(0, min(RAW_WIDTH, max(x1, x2)))
    bottom = max(0, min(RAW_HEIGHT, max(y1, y2)))
    return {
        "id": box.get("id") or f"box_{uuid4().hex[:10]}",
        "x": left,
        "y": top,
        "w": max(0, right - left),
        "h": max(0, bottom - top),
    }


def scaled_box(box: dict, width: int, height: int) -> dict:
    sx = width / RAW_WIDTH
    sy = height / RAW_HEIGHT
    return {
        "x": round(box["x"] * sx, 3),
        "y": round(box["y"] * sy, 3),
        "w": round(box["w"] * sx, 3),
        "h": round(box["h"] * sy, 3),
    }


def all_boxes(state: dict) -> list[dict]:
    boxes = []
    for image in state["images"]:
        image_id = image["id"]
        for box in state["boxes_by_image"].get(image_id, []):
            boxes.append({**box, "image_id": image_id, "image_name": image["name"]})
    return boxes


def export_masks(state: dict) -> dict:
    boxes = all_boxes(state)
    plain_boxes = [{key: box[key] for key in ("x", "y", "w", "h")} for box in boxes]
    return {
        "source_resolution": [RAW_WIDTH, RAW_HEIGHT],
        "ui_mask": {
            "1280x720": plain_boxes,
            "544x306": [scaled_box(box, VIDEO_WIDTH, VIDEO_HEIGHT) for box in boxes],
            "306x306": [scaled_box(box, RESIZED_WIDTH, RESIZED_HEIGHT) for box in boxes],
        },
    }


def parse_data_url(data_url: str) -> tuple[bytes, str]:
    header, encoded = data_url.split(",", 1)
    mime = header.split(";")[0].replace("data:", "")
    extension = mimetypes.guess_extension(mime) or ".png"
    if extension == ".jpe":
        extension = ".jpg"
    return base64.b64decode(encoded), extension


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    state = load_state()
    boxes = all_boxes(state)
    export = export_masks(state)
    return jsonify(
        {
            "raw_size": [RAW_WIDTH, RAW_HEIGHT],
            "video_size": [VIDEO_WIDTH, VIDEO_HEIGHT],
            "resized_size": [RESIZED_WIDTH, RESIZED_HEIGHT],
            "images": state["images"],
            "boxes_by_image": state["boxes_by_image"],
            "union_boxes_1280x720": boxes,
            "union_boxes_544x306": export["ui_mask"]["544x306"],
            "union_boxes_306x306": export["ui_mask"]["306x306"],
        }
    )


@app.route("/api/export")
def export_json():
    payload = json.dumps(export_masks(load_state()), indent=2)
    return Response(
        payload,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=ui_mask.json"},
    )


@app.route("/api/images", methods=["POST"])
def upload_image():
    ensure_dirs()
    if request.is_json:
        payload = request.get_json()
        image_bytes, extension = parse_data_url(payload["data_url"])
        original_name = payload.get("name") or f"pasted_{int(time.time())}{extension}"
    else:
        upload = request.files["image"]
        extension = Path(upload.filename).suffix.lower()
        if extension not in ALLOWED_IMAGE_EXTENSIONS:
            return jsonify({"error": f"Unsupported image type: {extension}"}), 400
        image_bytes = upload.read()
        original_name = upload.filename

    if extension.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported image type: {extension}"}), 400

    image_id = f"img_{uuid4().hex[:10]}"
    filename = f"{image_id}{extension}"
    path = IMAGE_DIR / filename
    path.write_bytes(image_bytes)

    state = load_state()
    image_record = {"id": image_id, "name": original_name, "filename": filename}
    state["images"].append(image_record)
    state["boxes_by_image"][image_id] = []
    save_state(state)
    return jsonify(image_record)


@app.route("/api/images/<image_id>/boxes", methods=["PUT"])
def save_boxes(image_id: str):
    payload = request.get_json()
    state = load_state()
    if image_id not in {image["id"] for image in state["images"]}:
        return jsonify({"error": f"Unknown image id: {image_id}"}), 404
    state["boxes_by_image"][image_id] = [
        box for box in (normalized_box(item) for item in payload.get("boxes", [])) if box["w"] > 0 and box["h"] > 0
    ]
    save_state(state)
    return jsonify({"boxes": state["boxes_by_image"][image_id]})


@app.route("/api/videos", methods=["POST"])
def upload_video():
    ensure_dirs()
    upload = request.files["video"]
    extension = Path(upload.filename).suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": f"Unsupported video type: {extension}"}), 400
    video_id = f"video_{uuid4().hex[:10]}"
    filename = f"{video_id}{extension}"
    path = VIDEO_DIR / filename
    upload.save(path)
    return jsonify({"id": video_id, "name": upload.filename, "filename": filename, "url": f"/videos/{filename}"})


@app.route("/images/<filename>")
def serve_image(filename: str):
    return send_file(IMAGE_DIR / filename)


@app.route("/videos/<filename>")
def serve_video(filename: str):
    return send_file(VIDEO_DIR / filename)


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="127.0.0.1", port=5055, debug=True)
