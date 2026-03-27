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
SKIP_FRAMES   = 1       # run YOLO every frame — required for stable tracking with persist=True
DETECT_WIDTH  = 640
DETECT_HEIGHT = 360
SHOW_VIDEO    = os.getenv("SHOW_VIDEO", "true").lower() not in ("false", "0", "no")

# Model
MODEL_FILE   = "yolo12n.pt"
TRACKER_FILE = "custom-botsort.yaml"
USE_ONNX     = False

# Detection
CONF_THRESHOLD = 0.40   # lower → catches more people, higher → fewer false positives
IOU_THRESHOLD  = 0.7
DETECT_IMGSZ   = 800
MIN_BOX_AREA   = 1500   # ~39×39 px minimum at 640×360 — allows background people too

# Tracking
MAX_MISSING_FRAMES = 300   # drop track after N frames — matches tracker track_buffer

# Display
DISPLAY_SKIP = 2  # only update the window every N frames (reduces GUI overhead)

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


def check_zone_entry(tid, prev, curr, zone_poly, state):
    """
    Zone entry logic — only outside → inside counts as IN.
    Rules:
      - If tid is in already_inside list → skip, never count
      - If person's very first detection is inside the zone → add to already_inside, never count
      - If person was already counted → never count again
      - prev outside, curr inside → count IN, add to already_inside
    """
    is_inside = point_in_polygon(curr, zone_poly)

    # ── Temporary list check: person is currently inside the zone ─────────────
    if tid in state["already_inside"]:
        # Keep the set current (remove if they left)
        if not is_inside:
            state["already_inside"].discard(tid)
        return

    is_first = tid not in state["seen_tids"]
    state["seen_tids"].add(tid)

    # First detection inside zone — person was already there when video started
    if is_first and is_inside:
        state["already_inside"].add(tid)
        return

    # Already counted this person — never count twice
    if tid in state["counted_ids"]:
        if is_inside:
            state["already_inside"].add(tid)  # re-enter → suppress future checks
        return

    was_inside = point_in_polygon(prev, zone_poly)

    if not was_inside and is_inside:
        now = datetime.now()
        state["counted_ids"].add(tid)
        state["already_inside"].add(tid)
        state["last_crossing"][tid] = now
        state["count_in"] += 1
        print(f"[{now:%H:%M:%S}] Person {tid} → IN  (entered zone)")
        state["log_file"].write(f"[Id:{tid}, Dir:IN, {now:%Y-%m-%d %H:%M:%S}]\n")
        state["log_file"].flush()


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
    inside_ids = state["already_inside"]

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
        has_history = bool(state["history"][tid])
        prev = state["history"][tid][-1] if has_history else mid_pt
        state["history"][tid].append(mid_pt)

        # Only check crossing once we have a real previous position
        if has_history:
            check_zone_entry(tid, prev, mid_pt, zone_poly, state)
        else:
            # First ever detection — just register inside/outside state, never count
            is_inside = point_in_polygon(mid_pt, zone_poly)
            state["seen_tids"].add(tid)
            if is_inside:
                state["already_inside"].add(tid)


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
        state["already_inside"].discard(tid)


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
        "already_inside": set(), # TEMP LIST — tids currently inside zone, skip crossing check
        "counted_ids":    set(), # tids already counted — never count twice
        "seen_tids":      set(), # tids seen at least once — detects first-appearance-inside
        "boxes":          {},
        "boxes_conf":     {},    # tid → confidence of current-frame box (for overlap tie-break)
        "last_seen":      {},
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
