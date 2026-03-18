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
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Run headless (no display inside Docker)
ENV SHOW_VIDEO=false

CMD ["python", "detect_people.py"]
