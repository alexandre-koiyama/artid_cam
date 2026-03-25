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
SKIP_FRAMES   = 2       # run YOLO every N frames (1 = every frame, 2–3 = faster)
DETECT_WIDTH  = 640      # resize frame to this before processing
DETECT_HEIGHT = 360
SHOW_VIDEO    = os.getenv("SHOW_VIDEO", "true").lower() not in ("false", "0", "no")

# Model
MODEL_FILE   = "yolo12n.pt"
TRACKER_FILE = "custom-botsort.yaml"
USE_ONNX     = False    # requires `pip install onnx` — won't build on macOS 11

# Detection
CONF_THRESHOLD = 0.15   # lower → catches more people, higher → fewer false positives
IOU_THRESHOLD  = 0.7    # NMS overlap
DETECT_IMGSZ   = 384    # YOLO input size (384 = sweet spot for 640×360 input)
MIN_BOX_AREA   = 100    # ignore boxes smaller than this (px²)

# Tracking
MAX_MISSING_FRAMES     = 20    # drop track after N frames without detection
CROSSING_COOLDOWN_SECS = 10.0  # ignore re-crossings within this window

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

def side_of_line(pt, p1, p2):
    """Positive / negative tells which side of the directed line p1→p2 the point is on."""
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


def crosses_segment(prev, curr, p1, p2):
    """True when the step prev→curr intersects the finite segment p1→p2."""
    dx_seg, dy_seg = p2[0] - p1[0], p2[1] - p1[1]
    dx_step, dy_step = curr[0] - prev[0], curr[1] - prev[1]
    denom = dx_seg * dy_step - dy_seg * dx_step
    if denom == 0:
        return False
    dx0, dy0 = prev[0] - p1[0], prev[1] - p1[1]
    t = (dx0 * dy_step - dy0 * dx_step) / denom
    u = (dx0 * dy_seg - dy0 * dx_seg) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
#  CROSSING LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def check_crossing(tid, prev, curr, p1, p2, in_sign, state):
    """Check if person `tid` crossed the line. Updates `state` dict in-place."""
    s_prev = side_of_line(prev, p1, p2)
    s_curr = side_of_line(curr, p1, p2)

    if s_prev * s_curr >= 0:
        return  # same side — no crossing
    if not crosses_segment(prev, curr, p1, p2):
        return  # didn't hit the drawn segment

    now = datetime.now()
    last_ts, _ = state["last_crossing"].get(tid, (None, None))
    if last_ts and (now - last_ts).total_seconds() < CROSSING_COOLDOWN_SECS:
        return  # bounce filter

    direction = "IN" if np.sign(s_prev) == in_sign else "OUT"
    state["last_crossing"][tid] = (now, s_curr)
    state["count_in"] += direction == "IN"
    state["count_out"] += direction == "OUT"

    print(f"[{now:%H:%M:%S}] Person {tid} → {direction}")
    state["log_file"].write(f"[Id:{tid}, Dir:{direction}, {now:%Y-%m-%d %H:%M:%S}]\n")
    state["log_file"].flush()


# ──────────────────────────────────────────────────────────────────────────────
#  DRAWING
# ──────────────────────────────────────────────────────────────────────────────

YELLOW = (0, 255, 255)
BLUE   = (230, 0, 0)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
CYAN   = (78, 244, 245)


def draw_overlay(frame, state, p1, p2):
    """Draw bounding boxes, trails, counting line, and counters."""
    history = state["history"]
    boxes = state["boxes"]
    last_crossing = state["last_crossing"]

    for tid, positions in history.items():
        color = YELLOW if tid in last_crossing else BLUE
        pts = np.array(positions, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, 2)

    for tid, (x1, y1, x2, y2) in boxes.items():
        color = YELLOW if tid in last_crossing else BLUE
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.line(frame, p1, p2, CYAN, 2)
    cv2.putText(frame, f"IN:  {state['count_in']}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    cv2.putText(frame, f"OUT: {state['count_out']}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)


# ──────────────────────────────────────────────────────────────────────────────
#  PROCESS DETECTIONS
# ──────────────────────────────────────────────────────────────────────────────

def process_detections(results, p1, p2, in_sign, state):
    """Extract boxes from YOLO results, update history, check crossings."""
    if results[0].boxes.id is None:
        return

    boxes_xy = results[0].boxes.xyxy.int().cpu().tolist()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    for box, tid in zip(boxes_xy, track_ids):
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        state["boxes"][tid] = (x1, y1, x2, y2)
        state["last_seen"][tid] = state["frame_count"]

        if tid not in state["history"]:
            state["history"][tid] = deque(maxlen=10)

        mid_pt = (cx, cy)
        prev = state["history"][tid][-1] if state["history"][tid] else mid_pt
        state["history"][tid].append(mid_pt)

        # Probe 3 points: center, top, bottom of bounding box
        dy = prev[1] - cy
        for p_prev, p_curr in [
            (prev, mid_pt),
            ((prev[0], y1 + dy), (cx, y1)),
            ((prev[0], y2 + dy), (cx, y2)),
        ]:
            check_crossing(tid, p_prev, p_curr, p1, p2, in_sign, state)
            if tid in state["last_crossing"]:
                break


def cleanup_stale_tracks(state):
    """Remove tracks that haven't been seen recently."""
    fc = state["frame_count"]
    stale = [t for t, f in state["last_seen"].items()
             if fc - f > MAX_MISSING_FRAMES]
    for tid in stale:
        state["boxes"].pop(tid, None)
        state["history"].pop(tid, None)
        state["last_seen"].pop(tid, None)


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
    # ── Load line config ──────────────────────────────────────────────────────
    with open("line_config.json") as f:
        cfg = json.load(f)
    p1 = tuple(cfg["p1"])
    p2 = tuple(cfg["p2"])
    p_in = tuple(cfg.get("p_in", [0, 0]))

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

    # ── Scale line points to detection resolution ─────────────────────────────
    if orig_w > 0 and orig_h > 0:
        sx, sy = DETECT_WIDTH / orig_w, DETECT_HEIGHT / orig_h
        p1 = (int(p1[0] * sx), int(p1[1] * sy))
        p2 = (int(p2[0] * sx), int(p2[1] * sy))
        p_in = (int(p_in[0] * sx), int(p_in[1] * sy))
    in_sign = np.sign(side_of_line(p_in, p1, p2))

    # ── Video output ──────────────────────────────────────────────────────────
    out, use_ffmpeg = open_video_writer(fps)

    # ── Tracking state ────────────────────────────────────────────────────────
    log_file = open("crossing_log.txt", "a")
    state = {
        "history":        {},
        "last_crossing":  {},
        "boxes":          {},
        "last_seen":      {},
        "count_in":       0,
        "count_out":      0,
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
                process_detections(results, p1, p2, in_sign, state)
                cleanup_stale_tracks(state)

            # Draw overlay every frame (for output video), but only push
            # to the window every DISPLAY_SKIP frames (GUI is expensive)
            fc = state["frame_count"]
            need_display = SHOW_VIDEO and fc % DISPLAY_SKIP == 0

            if need_display or out is not None:
                draw_overlay(small, state, p1, p2)

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
        print(f"\nTotal — IN: {state['count_in']}  OUT: {state['count_out']}")


if __name__ == "__main__":
    run()
