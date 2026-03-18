# Camera stream URL.
# In Docker this is set via the RTSP_URL environment variable (see docker-compose.yml).
# When running locally it falls back to the hardcoded address below.
import os
rstp = os.getenv("RTSP_URL", "rtsp://alexandrek:1234knknkn@192.168.1.44:554/stream1")