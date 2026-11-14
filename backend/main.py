# backend/main.py
from pathlib import Path
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import uuid
import os
import subprocess

# ================= CSRNet EK İMPORTLAR =================
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.csrnet import CSRNet

# ======================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models"))).resolve()
UPLOADS_DIR = (BASE_DIR / "uploads").resolve()
PROCESSED_DIR = (BASE_DIR / "processed").resolve()

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

_default_origins = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ======================================================
# 🔥 MODEL REGISTRY
# ======================================================
MODEL_REGISTRY = {
    "best": str(MODELS_DIR / "best.pt"),
    "best2": str(MODELS_DIR / "best2.pt"),
    "csrnet_ab": str(MODELS_DIR / "csrnet_AB_best.pth"),
}

_MODEL_CACHE = {}


def get_model(model_name: str):
    model_name = model_name if model_name in MODEL_REGISTRY else "best"
    if model_name == "csrnet_ab":
        model_name = "best"

    if model_name not in _MODEL_CACHE:
        model_path = MODEL_REGISTRY[model_name]
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=503,
                detail=f"Model weight not found: {Path(model_path).name}. Place weights in backend/models/ (see README.md).",
            )
        print(f"📦 Model yükleniyor: {model_path}")
        _MODEL_CACHE[model_name] = YOLO(model_path)
        _MODEL_CACHE[model_name].to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return _MODEL_CACHE[model_name]


# -------------------------------
# DEFAULT YOLO MODEL (lazy-cached via get_model)
# -------------------------------
MODEL_PATH = MODEL_REGISTRY["best"]
if os.path.exists(MODEL_PATH) and os.getenv("SKIP_STARTUP_MODEL_LOAD", "0") != "1":
    _MODEL_CACHE["best"] = YOLO(MODEL_PATH)
    _MODEL_CACHE["best"].to("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# CSRNet CONFIG (SENİN DEĞERLERİN)
# ======================================================
CSRNET_PATCH_SIZE = 384
CSRNET_STRIDE = 192
CSRNET_VIDEO_FRAME_SKIP = 5

CSRNET_IMAGENET_MEAN = (0.485, 0.456, 0.406)
CSRNET_IMAGENET_STD = (0.229, 0.224, 0.225)

csrnet_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CSRNET_IMAGENET_MEAN, CSRNET_IMAGENET_STD),
])

CSRNET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CSRNET_MODEL = None
_CSRNET_WEIGHT_CACHE = {}


def csrnet_gaussian_weight(h, w, sigma=0.5):
    key = (h, w, float(sigma))
    if key in _CSRNET_WEIGHT_CACHE:
        return _CSRNET_WEIGHT_CACHE[key]

    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    d = np.sqrt(xx**2 + yy**2)
    g = np.exp(-(d**2) / (2 * sigma**2))
    wt = torch.from_numpy(g.astype(np.float32))
    _CSRNET_WEIGHT_CACHE[key] = wt
    return wt


def get_csrnet_model():
    global _CSRNET_MODEL
    if _CSRNET_MODEL is not None:
        return _CSRNET_MODEL

    weights_path = MODEL_REGISTRY["csrnet_ab"]
    if not os.path.exists(weights_path):
        raise HTTPException(
            status_code=503,
            detail=f"CSRNet weight not found: {Path(weights_path).name}. Place weights in backend/models/ (see README.md).",
        )

    m = CSRNet(load_pretrained_vgg=False).to(CSRNET_DEVICE)
    ckpt = torch.load(weights_path, map_location=CSRNET_DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval()

    _CSRNET_MODEL = m
    print("✅ CSRNet yüklendi:", weights_path)
    return _CSRNET_MODEL


@torch.no_grad()
def csrnet_patch_inference_on_pil(model, image_pil):
    model.eval()

    img_w, img_h = image_pil.size
    img_tensor = csrnet_to_tensor(image_pil).to(CSRNET_DEVICE)

    out_h = int(math.ceil(img_h / 8.0))
    out_w = int(math.ceil(img_w / 8.0))

    density_map = torch.zeros((out_h, out_w), device=CSRNET_DEVICE)
    norm_map = torch.zeros((out_h, out_w), device=CSRNET_DEVICE)

    tops = list(range(0, max(1, img_h - CSRNET_PATCH_SIZE + 1), CSRNET_STRIDE))
    lefts = list(range(0, max(1, img_w - CSRNET_PATCH_SIZE + 1), CSRNET_STRIDE))

    if tops[-1] != max(0, img_h - CSRNET_PATCH_SIZE):
        tops.append(max(0, img_h - CSRNET_PATCH_SIZE))
    if lefts[-1] != max(0, img_w - CSRNET_PATCH_SIZE):
        lefts.append(max(0, img_w - CSRNET_PATCH_SIZE))

    for top in tops:
        for left in lefts:
            patch = img_tensor[:, top:top + CSRNET_PATCH_SIZE, left:left + CSRNET_PATCH_SIZE]

            ph, pw = patch.shape[1], patch.shape[2]
            if ph < CSRNET_PATCH_SIZE or pw < CSRNET_PATCH_SIZE:
                patch = F.pad(patch, (0, CSRNET_PATCH_SIZE - pw, 0, CSRNET_PATCH_SIZE - ph))

            pred = model(patch.unsqueeze(0))
            pred = torch.clamp(F.relu(pred), max=5.0)
            pred = pred.squeeze(0).squeeze(0)

            real_h = min(CSRNET_PATCH_SIZE, img_h - top)
            real_w = min(CSRNET_PATCH_SIZE, img_w - left)

            real_oh = int(math.ceil(real_h / 8.0))
            real_ow = int(math.ceil(real_w / 8.0))

            pred = pred[:real_oh, :real_ow]

            y0, x0 = top // 8, left // 8
            weight = csrnet_gaussian_weight(*pred.shape).to(CSRNET_DEVICE)

            y1 = min(out_h, y0 + pred.shape[0])
            x1 = min(out_w, x0 + pred.shape[1])

            density_map[y0:y1, x0:x1] += pred[:y1-y0, :x1-x0] * weight[:y1-y0, :x1-x0]
            norm_map[y0:y1, x0:x1] += weight[:y1-y0, :x1-x0]

    density_map = density_map / (norm_map + 1e-6)

    pred_low = density_map.cpu().numpy()
    pred_count = float(pred_low.sum())

    pred_vis = F.interpolate(
        density_map.unsqueeze(0).unsqueeze(0),
        size=(img_h, img_w),
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()

    return pred_count, pred_vis


def csrnet_overlay(bgr, pred_vis, alpha=0.45):
    dens = pred_vis.copy()
    dens[dens < 0] = 0
    if dens.max() > 0:
        dens /= dens.max()
    heat = cv2.applyColorMap((dens * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1.0, heat, alpha, 0)


# -------------------------------
# DURAK KOORDİNATLARI
# -------------------------------
STOP_COORDS = {
    "zincirlikuyu": {"name": "Zincirlikuyu Metrobüs", "lat": 41.067106, "lng": 29.011934},
    "mecidiyekoy": {"name": "Mecidiyeköy Metrobüs", "lat": 41.069895, "lng": 28.987237},
    "sogutlucesme": {"name": "Söğütlüçeşme Metrobüs", "lat": 40.990742, "lng": 29.024258},
    "avcilar": {"name": "Avcılar Metrobüs", "lat": 40.979167, "lng": 28.721944},

    "taksim": {"name": "Taksim Meydanı", "lat": 41.036948, "lng": 28.985078},
    "levent": {"name": "Levent", "lat": 41.081989, "lng": 29.019087},
    "yenikapi": {"name": "Yenikapı", "lat": 41.005321, "lng": 28.9552},
    "uskudar": {"name": "Üsküdar", "lat": 41.025641, "lng": 29.013941},

    "besiktas": {"name": "Beşiktaş İskele", "lat": 41.043008, "lng": 29.002762},
    "kadikoy_rihtim": {"name": "Kadıköy Rıhtım", "lat": 40.990217, "lng": 29.028069},
    "eminonu": {"name": "Eminönü", "lat": 41.016388, "lng": 28.971964},
    "karakoy": {"name": "Karaköy", "lat": 41.022823, "lng": 28.973426},

    "esenler_otogar": {"name": "Esenler Otogar", "lat": 41.046404, "lng": 28.899456},
    "cevahir_avm": {"name": "Cevahir AVM", "lat": 41.06051, "lng": 28.987472},
    "istiklal": {"name": "İstiklal Caddesi", "lat": 41.035137, "lng": 28.976875},
}

STOP_DENSITY = {stop_id: {"people": 0.0} for stop_id in STOP_COORDS.keys()}


def update_density(stop: str, people: float):
    STOP_DENSITY[stop]["people"] = float(people)


def calc_density_label(people: int) -> str:
    if people < 5:
        return "Düşük"
    elif people < 20:
        return "Orta"
    else:
        return "Yüksek"


def _safe_filename(name: str) -> str:
    base = Path(name or "upload.bin").name
    cleaned = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in base)
    return cleaned or "upload.bin"


def _resolve_ffmpeg() -> str:
    env_bin = os.getenv("FFMPEG_PATH")
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    which_bin = shutil.which("ffmpeg")
    if which_bin:
        return which_bin
    win_default = r"C:\ffmpeg\bin\ffmpeg.exe"
    if os.path.isfile(win_default):
        return win_default
    raise FileNotFoundError(
        "FFmpeg executable not found. Install FFmpeg on PATH or set FFMPEG_PATH in .env."
    )


# -------------------------------
# FOTOĞRAF ANALİZİ (AYNI)
# -------------------------------
@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    stop: str = Form(...),
    model_name: str = Form("best"),
):
    file_id = uuid.uuid4()
    safe_name = _safe_filename(file.filename)
    file_path = str(UPLOADS_DIR / f"{file_id}_{safe_name}")

    with open(file_path, "wb") as f:
        f.write(await file.read())

    if model_name == "csrnet_ab":
        image_pil = Image.open(file_path).convert("RGB")
        pred_count, pred_vis = csrnet_patch_inference_on_pil(
            get_csrnet_model(), image_pil
        )
        persons = int(round(pred_count))
        bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        annotated = csrnet_overlay(bgr, pred_vis)
    else:
        model = get_model(model_name)
        results = model(file_path)
        persons = 0
        if results and results[0].boxes:
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:
                    persons += 1
        annotated = results[0].plot()

    annotated_rel = f"processed/annotated_{file_id}.jpg"
    annotated_path = str(PROCESSED_DIR / f"annotated_{file_id}.jpg")
    cv2.imwrite(annotated_path, annotated)

    stop_info = STOP_COORDS.get(stop)
    lat = stop_info["lat"] if stop_info else None
    lng = stop_info["lng"] if stop_info else None

    if stop in STOP_DENSITY:
        update_density(stop, persons)

    return {
        "count": persons,
        "lat": lat,
        "lng": lng,
        "processed_image": f"/file/{annotated_rel}",
        "model_used": model_name,
    }


# -------------------------------
# VİDEO ANALİZİ (CSRNet + YOLO)
# -------------------------------
@app.post("/video")
async def analyze_video(
    video: UploadFile = File(...),
    stop: str = Form(...),
    model_name: str = Form("best"),
):
    file_id = uuid.uuid4()
    safe_name = _safe_filename(video.filename)
    input_path = str(UPLOADS_DIR / f"{file_id}_{safe_name}")

    with open(input_path, "wb") as f:
        f.write(await video.read())

    temp_avi_path = str(PROCESSED_DIR / f"tmp_{file_id}.avi")
    final_mp4_rel = f"processed/processed_{file_id}.mp4"
    final_mp4_path = str(PROCESSED_DIR / f"processed_{file_id}.mp4")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_avi_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("❌ VideoWriter açılamadı (XVID)")

    total_count = 0
    frame_results = []
    frame_number = 0
    processed_frames = 0  # ✅ YENİ

    # ================= CSRNet VIDEO =================
    if model_name == "csrnet_ab":
        csrnet_model = get_csrnet_model()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % CSRNET_VIDEO_FRAME_SKIP != 0:
                frame_number += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)

            pred_count, pred_vis = csrnet_patch_inference_on_pil(
                csrnet_model, image_pil
            )

            persons = int(round(pred_count))
            total_count += persons
            processed_frames += 1  # ✅ YENİ
            frame_results.append({"frame": frame_number, "count": persons})

            annotated = csrnet_overlay(frame, pred_vis)
            annotated = cv2.resize(annotated, (width, height))
            annotated = annotated.astype(np.uint8)
            out.write(annotated)

            frame_number += 1

    # ================= YOLO VIDEO (AYNI) =================
    else:
        model = get_model(model_name)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            persons = 0
            if results and results[0].boxes:
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0:
                        persons += 1

            total_count += persons
            frame_results.append({"frame": frame_number, "count": persons})

            annotated = results[0].plot()
            if annotated.shape[2] == 4:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_RGBA2BGR)
            annotated = cv2.resize(annotated, (width, height))
            annotated = annotated.astype(np.uint8)

            out.write(annotated)
            frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # -------------------------------
    # AVI -> MP4 (FFmpeg)
    # -------------------------------
    ffmpeg_bin = _resolve_ffmpeg()
    try:
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i", temp_avi_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                final_mp4_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg STDOUT:")
        print(e.stdout.decode(errors="ignore"))
        print("❌ FFmpeg STDERR:")
        print(e.stderr.decode(errors="ignore"))
        raise RuntimeError("❌ FFmpeg MP4 oluşturamadı")
    # geçici AVI sil
    if not os.path.exists(final_mp4_path):
        raise RuntimeError("❌ FFmpeg MP4 oluşturamadı")

    if os.path.exists(temp_avi_path):
        os.remove(temp_avi_path)

    # -------------------------------
    # ORTALAMA HESABI
    # -------------------------------
    if model_name == "csrnet_ab":
        avg_people = round(total_count / max(processed_frames, 1))
    else:
        avg_people = round(total_count / max(frame_number, 1))

    stop_info = STOP_COORDS.get(stop)
    lat = stop_info["lat"] if stop_info else None
    lng = stop_info["lng"] if stop_info else None
    if stop in STOP_DENSITY:
        update_density(stop, avg_people)

    return {
        "total_count": total_count,
        "avg_people": avg_people,
        "frame_results": frame_results,
        "lat": lat,
        "lng": lng,
        "processed_video": f"/file/{final_mp4_rel}",
        "fps": fps,
        "model_used": model_name,
    }


# -------------------------------
# DİĞER ENDPOINTLER
# -------------------------------
@app.get("/api/bus-stop/{stop_id}/density")
async def get_bus_stop_density(stop_id: str):
    raw_people = float(STOP_DENSITY.get(stop_id, {"people": 0.0})["people"])
    people = int(round(raw_people))
    density = calc_density_label(people)

    return {
        "stop_id": stop_id,
        "people": people,
        "density": density,
    }


@app.get("/api/all-densities")
async def get_all_densities():
    result = []
    for stop_id, info in STOP_COORDS.items():
        raw_people = float(STOP_DENSITY.get(stop_id, {"people": 0.0})["people"])
        people = int(round(raw_people))
        result.append(
            {
                "stop_id": stop_id,
                "lat": info["lat"],
                "lng": info["lng"],
                "people": people,
            }
        )
    return result


@app.post("/api/reset-densities")
async def reset_densities():
    for k in STOP_DENSITY.keys():
        STOP_DENSITY[k]["people"] = 0.0
    return {"success": True}


_ALLOWED_MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4"}


@app.get("/file/{path:path}")
async def get_file(path: str):
    """
    Safely serve processed media artifacts from PROCESSED_DIR only.
    Blocks directory traversal, absolute paths, and access outside processed/.
    """
    normalized = (path or "").replace("\\", "/").lstrip("/")
    if normalized.startswith("processed/"):
        rel_part = normalized[len("processed/"):]
    else:
        raise HTTPException(status_code=403, detail="Access denied: only processed media can be served.")

    if not rel_part or "/" in rel_part or ".." in rel_part or ":" in rel_part:
        raise HTTPException(status_code=403, detail="Access denied: invalid media path.")

    candidate = (PROCESSED_DIR / rel_part).resolve()
    try:
        candidate.relative_to(PROCESSED_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path escapes media directory.")

    if candidate.suffix.lower() not in _ALLOWED_MEDIA_EXTENSIONS:
        raise HTTPException(status_code=403, detail="Access denied: unsupported file extension.")

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(str(candidate))
