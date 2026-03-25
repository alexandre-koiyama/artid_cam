# ─────────────────────────────────────────────────────────────────────────────
#  Artidoro Cam — people crossing detector
#  Base: Python 3.11 slim (no GUI needed inside container)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System packages needed by OpenCV (libGL, libGlib) and by ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer — only re-runs if requirements.txt changes)
COPY requirements.txt .
# Install torch CPU-only first (avoids downloading ~2.5 GB of CUDA libs unused on this machine)
RUN pip install --no-cache-dir \
        torch==2.2.2+cpu \
        torchvision==0.17.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt --no-deps
RUN pip install --no-cache-dir \
        numpy==2.2.6 \
        opencv-python==4.12.0.88 \
        ultralytics==8.4.21 \
        ultralytics-thop==2.0.18

# Copy the rest of the project
COPY . .

# Run headless (no display inside Docker)
ENV SHOW_VIDEO=false

CMD ["python", "detect_people.py"]
