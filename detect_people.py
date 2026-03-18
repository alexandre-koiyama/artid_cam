import os
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
os.environ.setdefault("QT_LOGGING_TO_CONSOLE", "0")   # belt-and-suspenders Qt silence

import cv2, json, sys, subprocess
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import deque
import rstp

# ─────────────────────────────────────────────
#  SETTINGS  (tweak these without reading code)
# ─────────────────────────────────────────────

SKIP_FRAMES            = 3      # run YOLO every N frames (higher = faster, less accurate)
DETECT_WIDTH           = 640    # detection frame width
DETECT_HEIGHT          = 360    # detection frame height
# When running in Docker there is no screen — set SHOW_VIDEO=false env var to disable
SHOW_VIDEO             = os.getenv("SHOW_VIDEO", "true").lower() not in ("false", "0", "no")
CROSSING_COOLDOWN_SECS = 10.0   # seconds to ignore re-crossings (bounce filter)

# yolo12s.pt → small (19 MB) — recommended  |  yolo26n.pt → nano (5 MB) — faster but less accurate
MODEL_FILE         = "yolo26n.pt"
TRACKER_FILE       = "custom-botsort.yaml"  # custom-botsort.yaml | bytetrack.yaml

CONF_THRESHOLD     = 0.25   # lower = catches more people (raise to reduce false positives)
IOU_THRESHOLD      = 0.75   # NMS overlap threshold
DETECT_IMGSZ       = 704    # YOLO inference size (higher = better for small/far people)
MIN_BOX_AREA       = 500    # skip boxes smaller than this (pixels²) — filters noise
MAX_MISSING_FRAMES = 20     # drop a track after N frames without detection

cv2.setUseOptimized(True)
cv2.setNumThreads(4)


# ─────────────────────────────────────────────
#  GEOMETRY
# ─────────────────────────────────────────────

def side_of_line(pt, p1, p2):
    """Signed area — positive/negative tells which side of line p1→p2 the point is on."""
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


def crosses_segment(prev, curr, p1, p2):
    """
    True if step prev→curr intersects the finite segment p1→p2.
    Uses parametric intersection: t (along segment) and u (along step)
    must both be in [0, 1] for a real crossing.
    """
    dx_seg, dy_seg   = p2[0] - p1[0],    p2[1] - p1[1]
    dx_step, dy_step = curr[0] - prev[0], curr[1] - prev[1]
    denom = dx_seg * dy_step - dy_seg * dx_step
    if denom == 0:
        return False  # parallel — no crossing
    dx0, dy0 = prev[0] - p1[0], prev[1] - p1[1]
    t = (dx0 * dy_step - dy0 * dx_step) / denom
    u = (dx0 * dy_seg  - dy0 * dx_seg)  / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


# ─────────────────────────────────────────────
#  CROSSING LOGIC
# ─────────────────────────────────────────────

def check_crossing(tid, prev, curr, p1, p2, in_sign, last_crossing, count_in, count_out):
    """
    Counts a crossing for person `tid` only if:
      1. They moved from one side of the segment to the other, AND
      2. CROSSING_COOLDOWN_SECS have passed since their last crossing.
    Returns updated (count_in, count_out, last_crossing).
    """
    s_prev = side_of_line(prev, p1, p2)
    s_curr = side_of_line(curr, p1, p2)

    if s_prev * s_curr >= 0:                             # same side — no crossing
        return count_in, count_out, last_crossing
    if not crosses_segment(prev, curr, p1, p2):          # didn't cross the drawn segment
        return count_in, count_out, last_crossing

    now = datetime.now()
    last_ts, _ = last_crossing.get(tid, (None, None))
    if last_ts and (now - last_ts).total_seconds() < CROSSING_COOLDOWN_SECS:
        return count_in, count_out, last_crossing        # too soon — ignore bounce

    # ✅ Valid crossing
    direction = "IN" if np.sign(s_prev) == in_sign else "OUT"
    last_crossing[tid] = (now, s_curr)
    count_in  += direction == "IN"
    count_out += direction == "OUT"

    print(f"[{now:%H:%M:%S}] Person {tid} → {direction}")
    with open("crossing_log.txt", "a") as f:
        f.write(f"[Id:{tid}, Dir:{direction}, {now:%Y-%m-%d %H:%M:%S}]\n")

    return count_in, count_out, last_crossing


# ─────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────

def draw_frame(frame, history, boxes, last_crossing, p1, p2, count_in, count_out):
    """Draws trails, bounding boxes, the counting line, and IN/OUT counters."""
    for tid, positions in history.items():
        color = (0, 255, 255) if tid in last_crossing else (230, 0, 0)  # yellow = crossed, blue = not yet
        cv2.polylines(frame, [np.array(positions, np.int32).reshape(-1, 1, 2)], False, color, 2)

    for tid, (x1, y1, x2, y2) in boxes.items():
        color = (0, 255, 255) if tid in last_crossing else (230, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.line(frame, p1, p2, (0, 255, 0), 2)
    cv2.putText(frame, f"IN:  {count_in}",  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {count_out}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    # ── Line config ───────────────────────────────────────────────────────────
    cfg  = json.load(open("line_config.json"))
    p1   = tuple(cfg["p1"])
    p2   = tuple(cfg["p2"])
    p_in = tuple(cfg.get("p_in", [0, 0]))

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading YOLO model...")
    model = YOLO(MODEL_FILE)
    model.fuse()

    # ── Video source ──────────────────────────────────────────────────────────
    src = sys.argv[1] if len(sys.argv) > 1 else rstp.rstp
    if isinstance(src, str):
        src = src.strip().strip("'\"").rstrip(".").replace("rstp://", "rtsp://")
        if src.isdigit():
            src = int(src)
    cap   = cv2.VideoCapture(src)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Monitoring: {src}")

    # ── Scale line points to detection frame size ──────────────────────────────
    if orig_w > 0 and orig_h > 0:
        sx, sy = DETECT_WIDTH / orig_w, DETECT_HEIGHT / orig_h
        p1   = (int(p1[0]   * sx), int(p1[1]   * sy))
        p2   = (int(p2[0]   * sx), int(p2[1]   * sy))
        p_in = (int(p_in[0] * sx), int(p_in[1] * sy))
    in_sign = np.sign(side_of_line(p_in, p1, p2))

    # ── Optional video output (via FFmpeg, fallback to OpenCV) ────────────────
    out, use_ffmpeg, save_size = None, False, None
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        save_size   = (int(orig_w * 0.5), int(orig_h * 0.5))
        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
        try:
            out = subprocess.Popen([
                "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{save_size[0]}x{save_size[1]}", "-pix_fmt", "bgr24",
                "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", output_path,
            ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            use_ffmpeg = True
        except Exception:
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), save_size)
        print(f"Saving video to {output_path}")

    # ── Tracking state ────────────────────────────────────────────────────────
    history       = {}  # tid → deque of (cx, cy)  — movement trail
    last_crossing = {}  # tid → (datetime, side)   — last confirmed crossing
    current_boxes = {}  # tid → (x1, y1, x2, y2)  — latest bounding box
    last_seen     = {}  # tid → frame_count         — for stale track cleanup
    count_in, count_out, frame_count = 0, 0, 0

    if SHOW_VIDEO:
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection", 1280, 720)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        small = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT))

        # Run YOLO every SKIP_FRAMES to save CPU
        if frame_count % SKIP_FRAMES == 0:
            results = model.track(
                small, persist=True, classes=[0], verbose=False, device="cpu",
                tracker=TRACKER_FILE, imgsz=DETECT_IMGSZ,
                conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, agnostic_nms=True,
            )

            if results[0].boxes.id is not None:
                for box, tid in zip(
                    results[0].boxes.xyxy.int().cpu().tolist(),
                    results[0].boxes.id.int().cpu().tolist(),
                ):
                    x1, y1, x2, y2 = box
                    if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                        continue  # skip tiny boxes (noise / shadows)

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    current_boxes[tid] = (x1, y1, x2, y2)
                    last_seen[tid]     = frame_count

                    if tid not in history:
                        history[tid] = deque(maxlen=10)

                    # 3 probe points: top / center / bottom of the bounding box.
                    # If ANY crosses the line it counts — catches head, waist, or feet.
                    mid_pt = (cx, cy)
                    prev   = history[tid][-1] if history[tid] else mid_pt
                    history[tid].append(mid_pt)

                    dy = prev[1] - cy  # vertical offset from last frame
                    for p_prev, p_curr in [
                        (prev,               mid_pt),     # center
                        ((prev[0], y1 + dy), (cx, y1)),   # top
                        ((prev[0], y2 + dy), (cx, y2)),   # bottom
                    ]:
                        count_in, count_out, last_crossing = check_crossing(
                            tid, p_prev, p_curr, p1, p2, in_sign,
                            last_crossing, count_in, count_out,
                        )
                        if tid in last_crossing:
                            break  # already counted — skip remaining probe points

            # Remove stale tracks
            for tid in [t for t, f in last_seen.items() if frame_count - f > MAX_MISSING_FRAMES]:
                current_boxes.pop(tid, None)
                history.pop(tid, None)
                last_seen.pop(tid, None)

        # Draw & display
        if SHOW_VIDEO:
            draw_frame(small, history, current_boxes, last_crossing, p1, p2, count_in, count_out)
            cv2.imshow("Detection", small)

        # Write to output video if requested
        if out is not None:
            resized = cv2.resize(small, save_size)
            out.stdin.write(resized.tobytes()) if use_ffmpeg else out.write(resized)

        if SHOW_VIDEO and cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if out is not None:
        (out.stdin.close() or out.wait()) if use_ffmpeg else out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
