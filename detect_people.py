"""
People-crossing detector — counts IN / OUT crossings of a drawn line.
Uses YOLO (ultralytics) for detection + tracking, OpenCV for display.
"""

import os

os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
os.environ.setdefault("QT_LOGGING_TO_CONSOLE", "0")

import cv2
import json
import sys
import subprocess
import numpy as np
from pathlib import Path
from threading import Thread
from ultralytics import YOLO
from datetime import datetime
from collections import deque
import rstp

# ──────────────────────────────────────────────────────────────────────────────
#  SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

# Video
SKIP_FRAMES   = 2       # process every 2nd frame (good balance of speed vs accuracy)
DETECT_WIDTH  = 640
DETECT_HEIGHT = 360
SHOW_VIDEO    = os.getenv("SHOW_VIDEO", "true").lower() not in ("false", "0", "no")

# Model
# Options tested:
#   "yolo12n.pt"       — fastest, lower accuracy (nano)
#   "yolov8m.pt"       — slower, much better accuracy on crowded scenes (medium)
MODEL_FILE   = "yolov8n.pt"
TRACKER_FILE = "custom-botsort.yaml"
USE_ONNX     = False

# Detection
CONF_THRESHOLD = 0.35   # confidence threshold (0.0–1.0)
IOU_THRESHOLD  = 0.5    # NMS overlap threshold
DETECT_IMGSZ   = 800    # YOLO input resolution
MIN_BOX_AREA   = 500    # ignore tiny boxes (px²)

# Tracking
MAX_MISSING_FRAMES  = 90   # drop a lost track after ~3 s at 30 fps
EXIT_DEBOUNCE_FRAMES = 8   # frames outside the zone before exit is considered real (jitter filter)

# Display
DISPLAY_SKIP = 2  # refresh the window every 2nd frame

# Snapshots
SNAPSHOT_DIR = Path("snapshots")   # folder where crossing photos are saved
SNAPSHOT_DIR.mkdir(exist_ok=True)

# OpenCV threading
cv2.setUseOptimized(True)
cv2.setNumThreads(0)  # 0 = use all CPU cores


# ──────────────────────────────────────────────────────────────────────────────
#  THREADED VIDEO READER
# ──────────────────────────────────────────────────────────────────────────────

class ThreadedCapture:
    """Reads frames in a background thread so YOLO doesn't block on I/O."""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ok = True
        self.frame = None
        self._read_one()  # prime the first frame
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _read_one(self):
        self.ok, self.frame = self.cap.read()

    def _loop(self):
        while self.ok:
            self._read_one()

    def read(self):
        return self.ok, self.frame

    def get(self, prop):
        return self.cap.get(prop)

    def isOpened(self):
        return self.ok and self.cap.isOpened()

    def release(self):
        self.ok = False
        self.cap.release()


# ──────────────────────────────────────────────────────────────────────────────
#  ONNX AUTO-EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def get_model_path():
    """Return ONNX path if USE_ONNX is enabled, auto-exporting when needed."""
    if not USE_ONNX:
        return MODEL_FILE

    onnx_path = Path(MODEL_FILE).with_suffix(".onnx")
    if onnx_path.exists():
        return str(onnx_path)

    print(f"Exporting {MODEL_FILE} → ONNX (one-time, ~30 s)...")
    tmp = YOLO(MODEL_FILE)
    tmp.export(format="onnx", imgsz=DETECT_IMGSZ, simplify=True)
    print(f"Saved {onnx_path}")
    return str(onnx_path)


# ──────────────────────────────────────────────────────────────────────────────
#  GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
#  CROSSING LOGIC  (outside zone → inside zone = IN)
# ──────────────────────────────────────────────────────────────────────────────

def point_in_polygon(pt, poly):
    """Ray-casting test: True if pt is inside the polygon defined by poly."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def save_snapshot(tid, state):
    """Save a snapshot of the current frame when a person crosses into the zone."""
    frame = state.get("current_frame")
    if frame is None:
        return

    now = datetime.now()
    ts  = now.strftime("%Y%m%d_%H%M%S")

    # Full-frame snapshot with overlay already drawn
    full_path = SNAPSHOT_DIR / f"crossing_id{tid}_{ts}_full.jpg"
    cv2.imwrite(str(full_path), frame)

    # Cropped snapshot around the person's bounding box (with padding)
    box = state["boxes"].get(tid)
    if box is not None:
        x1, y1, x2, y2 = box
        pad = 30
        h, w = frame.shape[:2]
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        crop_path = SNAPSHOT_DIR / f"crossing_id{tid}_{ts}_crop.jpg"
        cv2.imwrite(str(crop_path), crop)

    print(f"  📸 Snapshot saved → {full_path.name}")


def check_zone_entry(tid, curr, zone_poly, state):
    """
    Zone entry logic — only outside → inside counts as IN.

    Two explicit sets are maintained every frame:
      ids_inside       — people confirmed inside the polygon
      ids_outside      — people confirmed outside the polygon
      exit_counter     — frames a person has been outside since last being inside
                         (must exceed EXIT_DEBOUNCE_FRAMES to confirm they really left)

    Counting rules:
      1. Person first appears INSIDE  → added to ids_inside, never counted.
      2. Person first appears OUTSIDE → added to ids_outside.
      3. Person moves outside → inside and was confirmed outside → count IN once.
      4. Person moves inside → outside → start exit debounce counter, stay in ids_inside
         until EXIT_DEBOUNCE_FRAMES consecutive outside frames confirm real exit.
      5. Short jitter (< EXIT_DEBOUNCE_FRAMES frames outside) resets counter → not counted.
      6. Person already counted re-enters → never counted again.
    """
    is_inside = point_in_polygon(curr, zone_poly)
    is_first  = tid not in state["ids_inside"] and tid not in state["ids_outside"]

    if is_first:
        if is_inside:
            state["ids_inside"].add(tid)
            state["counted_ids"].add(tid)  # already inside → never count
        else:
            state["ids_outside"].add(tid)
        return

    if is_inside:
        # Reset exit debounce — person is back inside
        state["exit_counter"].pop(tid, None)

        if tid in state["ids_outside"]:
            # Confirmed transition: outside → inside
            state["ids_outside"].discard(tid)
            state["ids_inside"].add(tid)

            if tid not in state["counted_ids"]:
                now = datetime.now()
                state["counted_ids"].add(tid)
                state["last_crossing"][tid] = now
                state["count_in"] += 1
                print(f"[{now:%H:%M:%S}] Person {tid} → IN (entered zone)")
                state["log_file"].write(f"[Id:{tid}, Dir:IN, {now:%Y-%m-%d %H:%M:%S}]\n")
                state["log_file"].flush()
                save_snapshot(tid, state)
        # else: still inside — nothing to do

    else:
        # Person is outside — run the debounce before confirming exit
        if tid in state["ids_inside"]:
            count = state["exit_counter"].get(tid, 0) + 1
            if count >= EXIT_DEBOUNCE_FRAMES:
                # Real exit confirmed
                state["exit_counter"].pop(tid, None)
                state["ids_inside"].discard(tid)
                state["ids_outside"].add(tid)
            else:
                # Still within jitter window — keep them as inside
                state["exit_counter"][tid] = count


# ──────────────────────────────────────────────────────────────────────────────
#  DRAWING
# ──────────────────────────────────────────────────────────────────────────────

YELLOW  = (0, 255, 255)
BLUE    = (230, 0, 0)
GREEN   = (0, 255, 0)
RED     = (0, 0, 255)
CYAN    = (78, 244, 245)
ORANGE  = (0, 165, 255)


def draw_overlay(frame, state, zone_poly):
    """Draw bounding boxes, trails, zone polygon, and IN counter."""
    history = state["history"]
    boxes = state["boxes"]
    last_crossing = state["last_crossing"]
    inside_ids = state["ids_inside"]

    for tid, positions in history.items():
        color = YELLOW if tid in last_crossing else (ORANGE if tid in inside_ids else BLUE)
        pts = np.array(positions, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, 2)

    for tid, (x1, y1, x2, y2) in boxes.items():
        color = YELLOW if tid in last_crossing else (ORANGE if tid in inside_ids else BLUE)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, color, -1)
        cv2.putText(frame, f"{tid}", (cx + 13, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Draw filled semi-transparent zone, then solid outline
    poly_arr = np.array(zone_poly, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly_arr], (0, 255, 0))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [poly_arr], True, GREEN, 2)

    cv2.putText(frame, f"IN: {state['count_in']}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)


# ──────────────────────────────────────────────────────────────────────────────
#  PROCESS DETECTIONS
# ──────────────────────────────────────────────────────────────────────────────

def process_detections(results, zone_poly, state):
    """Extract boxes from YOLO results, update history, check zone entry."""
    if results[0].boxes.id is None:
        return

    boxes_xy = results[0].boxes.xyxy.int().cpu().tolist()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    confs     = results[0].boxes.conf.cpu().tolist()

    # Clear current-frame boxes so stale positions from prior frames don't persist.
    # This prevents a person's dot from "jumping" to another person's position
    # when they overlap and the tracker briefly swaps IDs.
    state["boxes"].clear()
    state["boxes_conf"].clear()

    for box, tid, conf in zip(boxes_xy, track_ids, confs):
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # If two detections share a tid this frame, keep the higher-confidence one
        if tid in state["boxes"]:
            prev_box = state["boxes"][tid]
            prev_conf = state["boxes_conf"].get(tid, 0.0)
            if conf <= prev_conf:
                continue
        state["boxes"][tid] = (x1, y1, x2, y2)
        state["boxes_conf"][tid] = conf
        state["last_seen"][tid] = state["frame_count"]

        if tid not in state["history"]:
            state["history"][tid] = deque(maxlen=30)

        mid_pt = (cx, cy)
        state["history"][tid].append(mid_pt)

        check_zone_entry(tid, mid_pt, zone_poly, state)


def cleanup_stale_tracks(state):
    """Remove tracks that haven't been seen recently."""
    fc = state["frame_count"]
    stale = [t for t, f in state["last_seen"].items()
             if fc - f > MAX_MISSING_FRAMES]
    for tid in stale:
        state["boxes"].pop(tid, None)
        state["boxes_conf"].pop(tid, None)
        state["history"].pop(tid, None)
        state["last_seen"].pop(tid, None)
        state["exit_counter"].pop(tid, None)
        state["ids_inside"].discard(tid)
        state["ids_outside"].discard(tid)


# ──────────────────────────────────────────────────────────────────────────────
#  VIDEO OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def open_video_writer(fps):
    """Open FFmpeg pipe (preferred) or OpenCV VideoWriter as fallback."""
    if len(sys.argv) < 3:
        return None, False

    path = sys.argv[2]
    size = (DETECT_WIDTH, DETECT_HEIGHT)
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
             "-s", f"{size[0]}x{size[1]}", "-pix_fmt", "bgr24",
             "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "fast",
             "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        print(f"Saving video → {path} (ffmpeg)")
        return proc, True
    except FileNotFoundError:
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), size)
        print(f"Saving video → {path} (opencv)")
        return writer, False


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run():
    # ── Load zone config ──────────────────────────────────────────────────────
    with open("line_config.json") as f:
        cfg = json.load(f)
    zone_poly = [tuple(p) for p in cfg["zone"]]

    # ── Load model (ONNX if available) ─────────────────────────────────────────────
    model_path = get_model_path()
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # ── Open video source ─────────────────────────────────────────────────────────
    src = sys.argv[1] if len(sys.argv) > 1 else rstp.rstp
    if isinstance(src, str):
        src = src.strip().strip("'\"" ).rstrip(".").replace("rstp://", "rtsp://")
        if src.isdigit():
            src = int(src)

    # Use threaded reader for RTSP streams (non-blocking), plain VideoCapture for files
    is_stream = isinstance(src, str) and src.startswith("rtsp")
    if is_stream:
        cap = ThreadedCapture(src)
    else:
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"Monitoring: {src}  ({orig_w}×{orig_h} @ {fps:.0f} fps)")

    # ── Scale zone polygon to detection resolution ───────────────────────────
    if orig_w > 0 and orig_h > 0:
        sx, sy = DETECT_WIDTH / orig_w, DETECT_HEIGHT / orig_h
        zone_poly = [(int(x * sx), int(y * sy)) for x, y in zone_poly]

    # ── Video output ──────────────────────────────────────────────────────────
    out, use_ffmpeg = open_video_writer(fps)

    # ── Tracking state ────────────────────────────────────────────────────────
    log_file = open("crossing_log.txt", "a")
    state = {
        "history":        {},
        "last_crossing":  {},    # tid → datetime of confirmed IN
        "ids_inside":     set(), # people confirmed inside the polygon
        "ids_outside":    set(), # people confirmed outside the polygon
        "exit_counter":   {},    # tid → consecutive frames outside (debounce)
        "counted_ids":    set(), # tids already counted — never count twice
        "boxes":          {},
        "boxes_conf":     {},
        "last_seen":      {},
        "current_frame":  None,  # latest resized frame, used for snapshots
        "count_in":       0,
        "frame_count":    0,
        "log_file":       log_file,
    }

    if SHOW_VIDEO:
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection", 1280, 720)

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            state["frame_count"] += 1
            small = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT))
            state["current_frame"] = small.copy()  # keep a clean copy for snapshots

            if state["frame_count"] % SKIP_FRAMES == 0:
                results = model.track(
                    small, persist=True, classes=[0], verbose=False,
                    device="cpu", tracker=TRACKER_FILE, imgsz=DETECT_IMGSZ,
                    conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, agnostic_nms=True,
                )
                process_detections(results, zone_poly, state)
                cleanup_stale_tracks(state)

            # Draw overlay every frame (for output video), but only push
            # to the window every DISPLAY_SKIP frames (GUI is expensive)
            fc = state["frame_count"]
            need_display = SHOW_VIDEO and fc % DISPLAY_SKIP == 0

            if need_display or out is not None:
                draw_overlay(small, state, zone_poly)

            if need_display:
                cv2.imshow("Detection", small)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if out is not None:
                if use_ffmpeg:
                    out.stdin.write(small.tobytes())
                else:
                    out.write(small)

    finally:
        log_file.close()
        if out is not None:
            if use_ffmpeg:
                out.stdin.close()
                out.wait()
            else:
                out.release()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal IN: {state['count_in']}")


if __name__ == "__main__":
    run()
