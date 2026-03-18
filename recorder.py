import os
import subprocess
import time
import threading
import glob
from datetime import datetime

def is_active():
    """
    Checks if the current time is within the allowed recording hours.
    This ensures we only record when we actually want to (e.g., during store hours).
    """
    current_hour = datetime.now().hour
    
    # Retrieve start and end hours from Environment Variables, defaulting to 6 AM and 10 PM (22)
    start = int(os.getenv("START_HOUR", 6))
    end = int(os.getenv("END_HOUR", 22))
    
    # Handle normal shifts (e.g., 6 AM to 10 PM) and overnight shifts (e.g., 10 PM to 6 AM)
    if start <= end:
        return start <= current_hour < end
    else:
        return current_hour >= start or current_hour < end

def manage_recordings(folder, max_files=20, interval=60):
    """
    Runs continuously in the background to automatically delete old videos.
    This prevents the hard drive from filling up completely.
    """
    while True:
        try:
            # Find all MP4 video files in the recording folder
            files = glob.glob(os.path.join(folder, "cam_*.mp4"))
            
            # If we have more files than our maximum allowed limit
            if len(files) > max_files:
                
                # Sort files by the time they were last modified (oldest files first)
                files.sort(key=os.path.getmtime)
                
                # Figure out how many extra files we have
                files_to_delete = len(files) - max_files
                
                # Delete the oldest files until we are continuously exactly at max_files
                for i in range(files_to_delete):
                    old_file = files[i]
                    try:
                        os.remove(old_file)
                        print(f"🗑️ Deleted old recording to save space: {old_file}")
                    except OSError as e:
                        print(f"❌ Error deleting {old_file}: {e}")
                        
        except Exception as e:
            print(f"❌ Error in manage_recordings: {e}")
        
        # Wait a minute before checking the folder again
        time.sleep(interval)

def run_recorder():
    """
    Main function to start recording the camera feed in manageable chunks.
    """
    # Get the camera stream URL from the environment
    url = os.getenv("RTSP_URL")
    if not url:
        print("❌ Error: RTSP_URL environment variable is not set.")
        return

    # Prepare our FFmpeg command to record the RTSP stream in chunks
    # This takes the live stream and cuts it into a new file every X seconds
    cmd = [
        "ffmpeg", 
        "-rtsp_transport", "tcp",                 # Use TCP connection for stability
        "-fflags", "+genpts",                     # Generate presentation timestamps
        "-use_wallclock_as_timestamps", "1",      # Use actual clock time for sync
        "-i", url,                                # Input stream URL
        "-c:v", "copy", "-c:a", "aac",            # Copy video directly without re-encoding, compress audio
        "-f", "segment",                          # Output format is segmented files
        "-segment_time", os.getenv("SEGMENT_TIME", "300"), # Length of each video file (default 300s = 5 mins)
        "-reset_timestamps", "1",                 # Reset timestamps to zero for each new chunk
        "-strftime", "1",                         # Allow formatting timestamps in file names
        "/app/Recording/cam_%Y%m%d_%H%M%S.mp4"    # Output location and filename structure
    ]

    print(f"📹 Recorder active ({os.getenv('START_HOUR', 6)}:00 to {os.getenv('END_HOUR', 22)}:00)")

    # ---------------------------------------------------------
    # BACKGROUND STORAGE MANAGER THREAD
    # ---------------------------------------------------------
    
    # Ensure our recording directory actually exists
    record_dir = "/app/Recording"
    os.makedirs(record_dir, exist_ok=True)
    
    # Start our background process that deletes old videos
    # daemon=True means this thread will automatically close when our main program ends
    manager_thread = threading.Thread(target=manage_recordings, args=(record_dir, 20, 60), daemon=True)
    manager_thread.start()

    # ---------------------------------------------------------
    # MAIN RECORDING LOOP
    # ---------------------------------------------------------
    
    while True:
        # Only record if we are inside our designated working hours
        if is_active():
            print(f"🚀 [{datetime.now():%H:%M:%S}] Starting recording process...")
            
            # This will run FFmpeg indefinitely (or until the stream fails)
            subprocess.run(cmd)
            
            # If FFmpeg crashes or stops for any reason, wait 10 seconds before restarting it
            time.sleep(10)  
        else:
            # If we are outside working hours, just rest and wait
            print(f"💤 [{datetime.now():%H:%M:%S}] Outside recording window. Sleeping for a minute...")
            time.sleep(60)

if __name__ == "__main__":
    run_recorder()
