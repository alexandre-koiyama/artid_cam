# Camera stream URL.
# In Docker this is set via the RTSP_URL environment variable (see docker-compose.yml).
# When running locally it falls back to the hardcoded address below.
import os
rstp = os.getenv("RTSP_URL", "rtsp://admin:12345knknkn@192.168.0.108:554/cam/realmonitor?channel=1&subtype=0")