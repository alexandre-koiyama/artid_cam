import cv2
import json
import sys
import rstp

# Global variables to store the points and frames
points = []
frame = None
clone = None

def draw_line(event, x, y, flags, param):
    """
    Mouse callback function to handle clicks on the video frame.
    1st click: Start of the line
    2nd click: End of the line
    3rd click: Click in the area where people are coming FROM (incoming side).
    """
    global points, frame, clone
    
    # Check if left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Only allow up to 3 points
        if len(points) < 3:
            points.append([x, y])
            
            # Draw a small red circle where the user clicked
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
            # If two points have been clicked, draw a green line connecting them
            if len(points) == 2:
                cv2.line(frame, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
                
                # Show instructions on screen for the 3rd point
                cv2.putText(frame, "Line drawn!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(frame, "Now click on the side where people COME FROM (Entrance)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # If the third point is clicked, draw a blue dot to indicate the incoming side
            if len(points) == 3:
                cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
                cv2.putText(frame, "INCOMING AREA DEFINED", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, "Press 's' to save!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update the window to show the new drawing
            cv2.imshow("Draw Line", frame)

def run():
    global frame, clone, points
    
    # ---------------------------------------------------------
    # 1. SETUP VIDEO SOURCE
    # ---------------------------------------------------------
    
    # Try to get the video source from the command line, otherwise use our rstp module
    src = sys.argv[1] if len(sys.argv) > 1 else rstp.rstp
    
    # Clean up the source string (e.g. fix typos like 'rstp://' to 'rtsp://')
    if isinstance(src, str):
        src = src.strip().strip("'\"").rstrip('.').replace("rstp://", "rtsp://")
        # If the source is just a number (like '0' for webcams), convert it to an integer
        if src.isdigit(): 
            src = int(src)

    print(f"Loading frame at 1 second from {src}...")
    cap = cv2.VideoCapture(src)
    
    # ---------------------------------------------------------
    # 2. GET A SAMPLE FRAME
    # ---------------------------------------------------------
    
    # Attempt to skip ahead 1 second (1000 milliseconds) to avoid an empty initial frame
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    success, frame = cap.read()

    # If skipping by milliseconds fails (which is common for live RTSP streams),
    # manually read and discard 30 frames to get a good visual frame
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(30):
            success, frame = cap.read()
            if not success: 
                break
            
    cap.release()
    
    if not success:
        print("Error: Failed to read from source!")
        return

    # Keep a pure copy of the frame so we can reset our drawings later
    clone = frame.copy()
    
    # ---------------------------------------------------------
    # 3. SETUP USER INTERFACE
    # ---------------------------------------------------------
    
    # Create a resizeable window and link our mouse click function to it
    cv2.namedWindow("Draw Line", cv2.WINDOW_NORMAL) 
    cv2.setMouseCallback("Draw Line", draw_line)

    print("\n-----------------------------------------")
    print("INSTRUCTIONS:")
    print("1. Click exactly TWO points on the image to draw your line.")
    print("2. Click a THIRD point on the side where people COME FROM.")
    print("3. Press 'r' to reset if you make a mistake.")
    print("4. Press 's' to save the line to line_config.json")
    print("5. Press 'q' to quit without saving.")
    print("-----------------------------------------\n")

    # ---------------------------------------------------------
    # 4. MAIN INTERACTIVE LOOP
    # ---------------------------------------------------------
    
    while True:
        cv2.imshow("Draw Line", frame)
        
        # Wait for 1 millisecond for a key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            # RESET: Copy the pristine image back, clear our points
            frame = clone.copy()
            points = []
            print("Reset points. Click 3 points again.")
            
        elif key == ord("s"):
            # SAVE: Ensure exactly 3 points were clicked
            if len(points) == 3:
                config_path = "/home/alexandre/Desktop/Projects/Artidoro_Cam/line_config.json"
                
                # Try to load existing config so we don't overwrite other settings
                try:
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                except Exception:
                    # If it doesn't exist or is broken, create a default dictionary
                    cfg = {"direction": "down"}
                
                # Update the config with our new line coordinates and the incoming area point
                cfg["p1"] = points[0]
                cfg["p2"] = points[1]
                cfg["p_in"] = points[2]
                
                # Save it back to the file
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=4)
                    
                print(f"✅ Saved securely to {config_path}")
                break
            else:
                print("⚠️ Please select exactly 3 points before saving! (Or press 'q' to quit)")
                
        elif key == ord("q"):
            # QUIT: simply break the loop
            print("Quit without saving.")
            break

    # Clean up the windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
