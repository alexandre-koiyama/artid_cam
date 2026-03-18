# 🐳 Docker Guide — Artidoro Cam

People crossing line detector, containerised and ready to run anywhere.

---

## 📋 Table of Contents

1. [How Docker works (plain English)](#1-how-docker-works-plain-english)
2. [Project files explained](#2-project-files-explained)
3. [Developer setup (your home computer)](#3-developer-setup-your-home-computer)
4. [Deploy to another computer](#4-deploy-to-another-computer)
5. [Daily commands cheatsheet](#5-daily-commands-cheatsheet)
6. [Configuration reference](#6-configuration-reference)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. How Docker works (plain English)

```
Your code + Python + all libraries
          │
          ▼
    ┌───────────┐
    │   IMAGE   │  ← a "frozen snapshot" built once from the Dockerfile
    └───────────┘
          │  run
          ▼
    ┌───────────┐
    │ CONTAINER │  ← a running instance of the image (like a mini computer)
    └───────────┘
          │  writes files to
          ▼
    ┌───────────┐
    │  VOLUMES  │  ← shared folders between container and your real machine
    └───────────┘
```

- **Image** — like a ZIP file with everything inside. Built once, run anywhere.
- **Container** — the image running live. You can stop/start/restart it.
- **Volume** — a bridge between the container and your real hard drive.  
  Files written inside the container (logs, config) appear on your machine too.

---

## 2. Project files explained

| File | What it does |
|------|-------------|
| `Dockerfile` | Recipe to build the image — installs Python, packages, copies code |
| `docker-compose.yml` | How to run the container — camera URL, volumes, restart policy |
| `.dockerignore` | Files to exclude from the image (venv, videos, cache) |
| `requirements.txt` | Exact Python package versions installed inside the container |
| `line_config.json` | The counting line you drew with `draw_line.py` — mounted as a volume |
| `crossing_log.txt` | Crossing events log — mounted as a volume so you can read it on the host |

---

## 3. Developer setup (your home computer)

### Step 1 — Install Docker

```bash
# Linux (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # allow running docker without sudo
newgrp docker                   # apply group change without logout
```

Verify it works:
```bash
docker --version        # should print Docker version 20+
docker compose version  # should print Docker Compose version 2+
```

---

### Step 2 — Draw the counting line

Before building the container, you need to draw the line **once** on your local machine.

```bash
# Activate your local Python environment
source venv/bin/activate

# Run the line drawing tool
python draw_line.py
```

**Instructions inside the tool:**
1. Click point **P1** (start of the line)
2. Click point **P2** (end of the line)
3. Click **P_in** (a point on the INSIDE — the side where people come FROM)
4. Press **`s`** to save → creates `line_config.json`
5. Press **`q`** to quit

> ⚠️ Do this before building the Docker image. The `line_config.json` file is
> shared with the container via a volume, so you can update it later without rebuilding.

---

### Step 3 — Configure the camera URL

Edit `docker-compose.yml` and change the `RTSP_URL` to your camera's address:

```yaml
environment:
  RTSP_URL: "rtsp://YOUR_USER:YOUR_PASS@YOUR_CAMERA_IP:554/stream1"
```

Your camera's RTSP URL is usually in the camera's manual or web interface.
Common formats:
- Hikvision: `rtsp://user:pass@192.168.1.X:554/Streaming/Channels/101`
- Dahua:     `rtsp://user:pass@192.168.1.X:554/cam/realmonitor?channel=1`
- Generic:   `rtsp://user:pass@192.168.1.X:554/stream1`

---

### Step 4 — Build the Docker image

```bash
cd /path/to/Artidoro_Cam
docker compose build
```

This takes ~5–10 minutes the first time (downloads Python, installs packages).
Subsequent builds are fast because Docker caches unchanged layers.

---

### Step 5 — Run the detector

```bash
# Start in background (recommended)
docker compose up -d

# OR start in foreground to see logs immediately
docker compose up
```

---

### Step 6 — Watch the logs

```bash
docker compose logs -f
```

You should see lines like:
```
[14:32:10] Person 3 → IN
[14:32:45] Person 7 → OUT
```

Press `Ctrl+C` to stop watching. The container keeps running.

---

### Step 7 — Stop the detector

```bash
docker compose down
```

The container stops but the image stays. Next `docker compose up -d` starts it again instantly.

---

### Updating the code

After editing `detect_people.py` or any other file:

```bash
docker compose down
docker compose build   # rebuild with the new code
docker compose up -d
```

> You do NOT need to rebuild if you only change `line_config.json` or `docker-compose.yml` —
> those are mounted as volumes and take effect immediately on the next `docker compose up -d`.

---

## 4. Deploy to another computer

### Option A — Save image to a file (no internet needed)

**On your home computer:**
```bash
# 1. Build the image
docker compose build

# 2. Find the image name
docker images
# Look for something like: artidoro_cam-detector

# 3. Save it to a .tar file
docker save artidoro_cam-detector -o artidoro_cam.tar
```

**Transfer the following files to the other computer** (USB / SCP / Google Drive):
```
artidoro_cam.tar        ← the Docker image (~2 GB)
docker-compose.yml      ← run configuration
line_config.json        ← counting line
crossing_log.txt        ← can be empty, just needs to exist
```

**On the other computer:**
```bash
# 1. Install Docker (same as Step 1 above)

# 2. Load the image
docker load -i artidoro_cam.tar

# 3. Edit the camera URL
nano docker-compose.yml

# 4. Create the log file if it doesn't exist
touch crossing_log.txt

# 5. Run
docker compose up -d

# 6. Check logs
docker compose logs -f
```

---

### Option B — Docker Hub (easiest for multiple machines)

**On your home computer:**
```bash
# 1. Create a free account at https://hub.docker.com

# 2. Login
docker login

# 3. Build and tag with your Docker Hub username
docker build -t YOUR_DOCKERHUB_USERNAME/artidoro_cam:latest .

# 4. Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/artidoro_cam:latest
```

**Update `docker-compose.yml` to pull from Hub instead of building locally:**
```yaml
services:
  detector:
    image: YOUR_DOCKERHUB_USERNAME/artidoro_cam:latest  # ← use this on remote machines
    # build: .                                           # ← keep this on your dev machine
```

**On the other computer — only 3 files needed:**
```
docker-compose.yml      ← edited to use image: instead of build:
line_config.json        ← counting line
crossing_log.txt        ← empty file (must exist)
```

```bash
# 1. Install Docker

# 2. Edit RTSP_URL in docker-compose.yml

# 3. Pull and run
docker compose up -d

# 4. Check logs
docker compose logs -f
```

**To push an update after changing the code:**
```bash
# On your home computer
docker build -t YOUR_DOCKERHUB_USERNAME/artidoro_cam:latest .
docker push YOUR_DOCKERHUB_USERNAME/artidoro_cam:latest

# On the remote computer
docker compose pull
docker compose up -d
```

---

### Option C — Copy files and rebuild (simplest)

```bash
# On your home computer — create a zip (excludes venv, videos, cache)
cd /home/alexandre/Desktop/Projects
zip -r artidoro_cam.zip Artidoro_Cam \
  --exclude "*/venv/*" \
  --exclude "*/VIDEOS_TEST/*" \
  --exclude "*/__pycache__/*" \
  --exclude "*.mp4"
```

Copy `artidoro_cam.zip` to the other machine, then:

```bash
unzip artidoro_cam.zip
cd Artidoro_Cam
docker compose build
docker compose up -d
```

---

## 5. Daily commands cheatsheet

```bash
# Start the detector (background)
docker compose up -d

# Stop the detector
docker compose down

# Restart after a config change
docker compose down && docker compose up -d

# Watch live logs
docker compose logs -f

# See if the container is running
docker compose ps

# Rebuild after code changes
docker compose build

# Rebuild and restart in one command
docker compose up -d --build

# Open a shell inside the running container (for debugging)
docker exec -it artidoro-detector bash

# Delete the image to free disk space (need to rebuild after this)
docker rmi artidoro_cam-detector
```

---

## 6. Configuration reference

All settings are in `docker-compose.yml` under `environment:`.
No need to touch the Python code to change these.

| Variable | Default | Description |
|----------|---------|-------------|
| `RTSP_URL` | *(camera IP)* | Full RTSP URL of the camera stream |
| `SHOW_VIDEO` | `false` | Show live window — always `false` in Docker |

Detection thresholds (edit `detect_people.py`, then rebuild):

| Constant | Default | Description |
|----------|---------|-------------|
| `CONF_THRESHOLD` | `0.25` | Min detection confidence. Lower = catches more people |
| `MIN_BOX_AREA` | `500` | Ignore boxes smaller than this (filters shadows/noise) |
| `CROSSING_COOLDOWN_SECS` | `10.0` | Seconds before same person can cross again |
| `SKIP_FRAMES` | `3` | Run YOLO every N frames (lower = more accurate, more CPU) |
| `MODEL_FILE` | `yolo26n.pt` | YOLO model file — swap for `yolo12s.pt` for better accuracy |

---

## 7. Troubleshooting

### Container exits immediately
```bash
docker compose logs
# Read the error at the bottom
```

Most common causes:
- **Can't open camera** — wrong `RTSP_URL`. Test the URL with VLC first:  
  `vlc rtsp://user:pass@camera_ip:554/stream1`
- **`line_config.json` not found** — run `draw_line.py` first and make sure the file exists
- **Port already in use** — another process is using the same port

---

### `docker compose` command not found
```bash
# Try the old syntax
docker-compose up -d
```

---

### Permission denied running docker
```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

### Image takes too long to build
Normal for first build (~5–10 min). After that, only changed layers rebuild.
Make sure you're not accidentally copying `venv/` or `VIDEOS_TEST/` — check `.dockerignore`.

---

### Check how much disk space Docker is using
```bash
docker system df
```

Clean up unused images/containers:
```bash
docker system prune
```
