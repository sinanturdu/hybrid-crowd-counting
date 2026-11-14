# Hybrid Crowd Counting & Public Transit Density Analysis (YOLOv8 + CSRNet)

A full-stack graduation project (*Bitirme Projesi*) for crowd counting, density estimation, and geospatial transit-stop visualization across Istanbul public transport hubs. The system combines object detection (**YOLOv8**) for sparse-to-moderate crowds and dilated convolutional density map regression (**CSRNet**) for high-density scenes, served through a **FastAPI** backend and an interactive **React + Vite + Tailwind CSS + Chart.js + Google Maps / deck.gl** web application.

> **Language:** [English (README.md)](README.md) | [Türkçe (docs/README.tr.md)](docs/README.tr.md)

---

## Repository Structure

```text
hybrid-crowd-counting/
├── backend/                        # Production FastAPI application
│   ├── main.py                     # API entry point (YOLOv8 + CSRNet inference, stop densities)
│   ├── models/
│   │   ├── csrnet.py               # PyTorch CSRNet (VGG16 front-end + dilated conv back-end)
│   │   └── __init__.py             # (Runtime weights best.pt, best2.pt, csrnet_AB_best.pth go here; git-ignored)
│   ├── tests/
│   │   └── test_security_and_api.py# Security regression & inference integration tests
│   ├── legacy/                     # Archived early Node.js/Express prototype & unused helpers
│   ├── uploads/                    # Runtime temporary upload directory (git-ignored)
│   ├── processed/                  # Runtime annotated image/video output directory (git-ignored)
│   ├── requirements.txt            # Direct pinned backend dependencies
│   ├── requirements-lock.txt       # Full exact environment snapshot (Python 3.12.7 + CUDA 12.1)
│   └── .env.example                # Backend environment variable template
├── frontend/                       # React 19 + Vite 7 + Tailwind CSS + Chart.js + Google Maps UI
│   ├── src/
│   │   ├── config.js               # Environment-driven API base URL & Google Maps configuration
│   │   ├── context/MapContext.jsx  # Stop marker & deck.gl heatmap state synchronization
│   │   ├── data/stops.js           # Coordinates & metadata for 15 Istanbul transit stops
│   │   ├── components/             # Navbar (dark mode) & Sidebar navigation
│   │   └── pages/                  # Dashboard, ImageAnalysis, VideoAnalysis, MapPage
│   ├── public/stops/               # Local transit-stop popup photos (git-ignored; see public/stops/README.md)
│   ├── package.json
│   ├── package-lock.json
│   └── .env.example                # Frontend environment variable template
├── research/                       # Machine learning training, evaluation & preprocessing scripts
│   ├── training/
│   │   ├── csrnet_shanghaitech/    # Operational PyTorch CSRNet (Part A + Part B joint training)
│   │   ├── csrnet_ucf_qnrf/        # CSRNet UCF-QNRF experiments
│   │   ├── csrnet_v1/              # Early CSRNet training prototype
│   │   └── yolo/                   # YOLOv8 & YOLOv9 training scripts
│   ├── evaluation/
│   │   ├── csrnet_shanghaitech/    # Patch-based evaluation & ShanghaiTech A+B metrics
│   │   ├── csrnet_ucf_qnrf/        # UCF-QNRF evaluation & patch inference scripts
│   │   ├── csrnet_v1/              # Early CSRNet evaluation scripts
│   │   └── yolo/                   # YOLO test scripts & sanitized training run logs/curves
│   ├── preprocessing/              # Dataset converters (CrowdHuman, MOT20, ShanghaiTech, UCF-QNRF)
│   ├── configs/                    # Dataset YAML definitions & dataset attribution records
│   ├── legacy/                     # Early Streamlit/RandomForest prototype & webcam scripts
│   └── requirements.txt            # Research environment dependencies
├── examples/                       # Place rights-cleared sample inputs here (see examples/README.md)
├── docs/                           # Detailed technical documentation
│   ├── README.tr.md                # Turkish project documentation
│   ├── ARCHITECTURE.md             # System architecture & API contract specification
│   ├── MODELS_AND_EXPERIMENTS.md   # Verified model provenance, metrics & reproducibility notes
│   └── SECURITY_AND_LICENSING.md   # Security hardening, Google Maps key guide & licensing audit
├── .env.example                    # Root environment variable reference
└── .gitignore                      # Excludes weights, datasets, unapproved media, venvs & secrets
```

---

## Key Features (Verified from Implementation)

1. **Multi-Model Image Crowd Counting (`/image` -> `POST /upload-image`):**
   - Upload an image, select one of 15 Istanbul transit hubs, and choose between three runtime models:
     - `best` — YOLOv8s trained on `reid.v1i.yolov8` (person detection).
     - `best2` — YOLOv8s trained for 80 epochs on local person detection dataset.
     - `csrnet_ab` — PyTorch CSRNet trained jointly on ShanghaiTech Part A + Part B using $384 \times 384$ overlapping patch inference ($\text{stride} = 192$) with 2D Gaussian blending and JET density-map overlay.
   - Displays the detected person count, renders the annotated bounding-box or heatmap image, and updates the selected transit stop's live density marker on the map.
2. **Video Crowd Analysis & Time-Series Synchronization (`/video` -> `POST /video`):**
   - Processes uploaded videos frame-by-frame with YOLOv8 (or every 5th frame with CSRNet), writes annotated frames, and transcodes the output to browser-compatible H.264 MP4 (`yuv420p`, `+faststart`) via FFmpeg.
   - Renders an interactive **Chart.js** line chart of per-frame counts synchronized with the HTML5 video player's current playback timestamp, plus a scrollable frame table and overall average count.
3. **Interactive Geospatial Transit Map (`/map`):**
   - Displays 15 major Istanbul transit hubs (Zincirlikuyu, Mecidiyeköy, Söğütlüçeşme, Avcılar, Taksim, Levent, Yenikapı, Üsküdar, Beşiktaş, Kadıköy Rıhtım, Eminönü, Karaköy, Esenler Otogar, Cevahir AVM, İstiklal Caddesi) using Google Maps Advanced Markers and a toggleable **deck.gl `HeatmapLayer`**.
   - Polls `GET /api/all-densities` every 10 seconds and queries `GET /api/bus-stop/{stop_id}/density` when a stop marker is clicked.
4. **Dashboard (`/`):**
   - Displays a sample Chart.js bar chart overview (note: currently uses static demonstration values; see [Limitations](#known-limitations)).

---

## Quick Start & Reproducible Setup

### Prerequisites

| Component | Verified Version | Notes |
|---|---|---|
| **OS** | Windows 10/11 or Linux | Tested on Windows with NVIDIA RTX 3050 Ti Laptop GPU |
| **Python** | **3.12.7** (compatible with 3.10–3.12) | Original backend venv built with Python 3.12.7 |
| **PyTorch / CUDA** | **torch 2.5.1 + torchvision 0.20.1 (CUDA 12.1)** | CPU fallback (`torch.cuda.is_available()`) is built in |
| **Ultralytics** | **8.3.237** | Used for loading `best.pt` and `best2.pt` |
| **FFmpeg** | Any recent build with `libx264` | Must be on system `PATH` or set via `FFMPEG_PATH` |
| **Node.js / npm** | **Node.js 20+ / 24.x**, **npm 10+ / 11.x** | Tested with Node v24.11.1, npm 11.6.2 |

### 1. Place Runtime Model Weights

Model weights are excluded from Git (`~105 MB` total) due to size and dataset licensing considerations. Place the three verified weight files in `backend/models/`:

| Model Key | File Path | Size | SHA-256 / MD5 (first 10) | Architecture & Source Run |
|---|---|---|---|---|
| `best` | `backend/models/best.pt` | 22.5 MB | MD5 `A3A32B3DF5` | YOLOv8s (`runs/detect/train14/weights/best.pt`) |
| `best2` | `backend/models/best2.pt` | 22.5 MB | MD5 `CB238468C7` | YOLOv8s (`yolo8/runs/detect/train/weights/best2.pt`) |
| `csrnet_ab` | `backend/models/csrnet_AB_best.pth` | 65.1 MB | MD5 `F34C96BDCE` | PyTorch CSRNet (`ShanghaiTech/weights/csrnet_AB_best.pth`) |

See [docs/MODELS_AND_EXPERIMENTS.md](docs/MODELS_AND_EXPERIMENTS.md) for full provenance and training metrics.

> **Clean-Clone Behavior (Without Local Assets):**
> - If a fresh clone is started before placing the model weights in `backend/models/`, the backend server starts normally (`GET /api/all-densities` and `GET /api/bus-stop/{id}/density` work immediately), and inference endpoints (`POST /upload-image`, `POST /video`) return a clear `503 Service Unavailable` JSON message instructing you to place the weights in `backend/models/`.
> - Local transit-stop reference photographs in `frontend/public/stops/*.jpg` are also excluded from Git due to unconfirmed redistribution rights. On a clean clone, map marker popups automatically hide the missing image element (via a DOM `error` event listener) while still displaying the stop title, density badge, and live person count cleanly. To use custom stop photos locally, place `<stop_id>.jpg` files in `frontend/public/stops/` (matching the `image` paths in `frontend/src/data/stops.js`).

### 2. Backend Setup (FastAPI)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate

# (Recommended for NVIDIA GPUs) Install CUDA 12.1 PyTorch first:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining backend dependencies:
pip install -r requirements.txt

# Start the FastAPI server on port 8000:
uvicorn main:app --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### 3. Frontend Setup (React + Vite)

```powershell
cd frontend
cp .env.example .env
# Edit .env and set VITE_GOOGLE_MAPS_API_KEY to your HTTP-referrer-restricted browser key

npm ci
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Security Notes (Important)

- **Frontend Environment Variables Are Public:** Variables prefixed with `VITE_` (such as `VITE_GOOGLE_MAPS_API_KEY`) are embedded into the compiled browser bundle by Vite. Using `.env` keeps keys out of Git commits, **not** out of the browser. You **must** restrict your Google Maps key in Google Cloud Console by **HTTP referrer** and limit its API scope exclusively to **Maps JavaScript API**. See [docs/SECURITY_AND_LICENSING.md](docs/SECURITY_AND_LICENSING.md).
- **Safe Media Serving:** `GET /file/{path}` strictly serves files resolved inside `backend/processed/` with allowed media extensions (`.jpg`, `.jpeg`, `.png`, `.mp4`), rejecting any path traversal (`..`), absolute paths, or access to `uploads/`, `models/`, or source code with HTTP `403`.
- **Restricted CORS:** Wildcard CORS (`*`) has been replaced with an explicit allowlist configured via `CORS_ALLOWED_ORIGINS` (defaulting to `localhost:5173` and `localhost:3000` for local development).

---

## Known Limitations

- **In-Memory Stop Densities:** Transit stop counts (`STOP_DENSITY`) are stored in process memory and reset when the backend restarts or when `POST /api/reset-densities` is called.
- **Static Dashboard Overview:** `src/pages/Dashboard.jsx` currently displays static demonstration bar-chart values rather than live database aggregations.
- **CSRNet Video Frame Skipping:** To keep CPU/GPU runtime tractable, `main.py` evaluates every 5th frame (`CSRNET_VIDEO_FRAME_SKIP = 5`) during CSRNet video analysis and writes only evaluated frames to the output video.
- **Experimental Evaluation Caveats:** While `train14` (`best.pt`), `yolo8/train` (`best2.pt`), and `ShanghaiTech` CSRNet (`MAE 35.84`, `RMSE 69.92`) have valid train/val splits, two auxiliary research runs (`train15` and `train16`) evaluated on training images (`val: images/train` in `data_MOT20_YOLO.yaml`) and their `0.986 mAP50` scores should not be cited as held-out generalization metrics. See [docs/MODELS_AND_EXPERIMENTS.md](docs/MODELS_AND_EXPERIMENTS.md).

---

## Documentation Index

- [Türkçe Dokümantasyon (docs/README.tr.md)](docs/README.tr.md)
- [Architecture & API Specification (docs/ARCHITECTURE.md)](docs/ARCHITECTURE.md)
- [Models, Experiments & Excluded Assets Inventory (docs/MODELS_AND_EXPERIMENTS.md)](docs/MODELS_AND_EXPERIMENTS.md)
- [Security, Google Maps Key Hardening & Licensing Review (docs/SECURITY_AND_LICENSING.md)](docs/SECURITY_AND_LICENSING.md)
