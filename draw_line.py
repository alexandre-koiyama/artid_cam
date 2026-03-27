import cv2
import json
import sys
import numpy as np
import rstp

# ─────────────────────────────────────────────────────────────────────────────
#  INSTRUCTIONS
#  Click to place polygon points around the entry zone.
#  Need at least 3 points. Right-click or press 'c' to close the polygon.
#  Press 's' to save, 'r' to reset, 'q' to quit.
# ─────────────────────────────────────────────────────────────────────────────

ZONE_COLOR  = (0, 255, 0)    # green
DOT_COLOR   = (0, 0, 255)    # red
FILL_ALPHA  = 0.20

points = []
frame  = None
clone  = None
closed = False


def redraw(img, pts, is_closed):
    """Redraw the polygon on img."""
    # Fill if closed
    if is_closed and len(pts) >= 3:
        poly_arr = np.array(pts, dtype="int32")
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly_arr], ZONE_COLOR)
        cv2.addWeighted(overlay, FILL_ALPHA, img, 1 - FILL_ALPHA, 0, img)
        cv2.polylines(img, [poly_arr], True, ZONE_COLOR, 2)
    else:
        # Draw edges so far
        for i in range(1, len(pts)):
            cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]), ZONE_COLOR, 2)
        # Dashed closing edge preview (just draw it faintly)
        if len(pts) >= 3:
            cv2.line(img, tuple(pts[-1]), tuple(pts[0]), (100, 200, 100), 1)

    # Dots
    for i, p in enumerate(pts):
        cv2.circle(img, tuple(p), 6, DOT_COLOR, -1)
        cv2.putText(img, str(i + 1), (p[0] + 8, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, DOT_COLOR, 1)

    # Instructions
    if not is_closed:
        n = len(pts)
        if n == 0:
            hint = "Click to place zone corners"
        elif n < 3:
            hint = f"Point {n}/3+ — keep clicking to add corners"
        else:
            hint = f"{n} points — click more OR right-click / press 'c' to close"
        cv2.putText(img, hint, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        cv2.putText(img, f"Zone ready ({len(pts)} pts). Press 's' to save or 'r' to reset.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_COLOR, 2)


def on_mouse(event, x, y, flags, param):
    global points, frame, clone, closed
    if closed:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        frame = clone.copy()
        redraw(frame, points, closed)
        cv2.imshow("Draw Zone", frame)
    elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= 3:
        closed = True
        frame = clone.copy()
        redraw(frame, points, closed)
        cv2.imshow("Draw Zone", frame)


def run():
    global frame, clone, points, closed

    # ── Video source ──────────────────────────────────────────────────────────
    src = sys.argv[1] if len(sys.argv) > 1 else rstp.rstp
    if isinstance(src, str):
        src = src.strip().strip("'\"").rstrip(".").replace("rstp://", "rtsp://")
        if src.isdigit():
            src = int(src)

    print(f"Loading frame from {src}...")
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(30):
            success, frame = cap.read()
            if success:
                break
    cap.release()

    if not success:
        print("Error: could not read a frame from source.")
        return

    clone = frame.copy()
    redraw(frame, points, closed)

    cv2.namedWindow("Draw Zone", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw Zone", on_mouse)

    print("\n-----------------------------------------")
    print("INSTRUCTIONS:")
    print("  Left-click    → add a corner to the zone polygon")
    print("  Right-click   → close polygon (min 3 points)")
    print("  'c'           → close polygon")
    print("  'r'           → reset all points")
    print("  's'           → save zone to line_config.json")
    print("  'q'           → quit without saving")
    print("-----------------------------------------\n")

    config_path = "line_config.json"

    while True:
        cv2.imshow("Draw Zone", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            points = []
            closed = False
            frame = clone.copy()
            redraw(frame, points, closed)
            print("Reset — click to place zone corners again.")

        elif key == ord("c") and len(points) >= 3 and not closed:
            closed = True
            frame = clone.copy()
            redraw(frame, points, closed)
            print(f"Polygon closed with {len(points)} points. Press 's' to save.")

        elif key == ord("s"):
            if closed and len(points) >= 3:
                cfg = {"zone": points}
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=4)
                print(f"✅ Saved zone ({len(points)} points) to {config_path}")
                break
            else:
                if not closed:
                    print(f"⚠️  Close the polygon first (right-click or 'c'). Have {len(points)} points.")
                else:
                    print("⚠️  Need at least 3 points.")

        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
