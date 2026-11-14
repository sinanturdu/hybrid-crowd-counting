from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import tempfile
import os
from typing import List, Dict


# -------------------------------------------------
# FastAPI uygulaması
# -------------------------------------------------
app = FastAPI(title="Bitirme Projesi - People Counter API")

# React için CORS izni
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Modeli yükle (GPU varsa GPU'ya al)
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Kullanılan cihaz:", DEVICE)

# Eğittiğin en iyi modeli kullan
# Gerekirse burayı değiştirirsin:
MODEL_PATH = r"runs/detect/train14/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")

model = YOLO(MODEL_PATH).to(DEVICE)


# -------------------------------------------------
# Yardımcı fonksiyon: tek frame'de insan say
# -------------------------------------------------
def count_people_in_frame(frame: np.ndarray, conf: float = 0.5) -> int:
    """
    Verilen frame üzerinde YOLO ile 'person' sınıfını sayar.
    """
    results = model(
        frame,
        conf=conf,
        iou=0.5,
        classes=[0],   # 0 = person
        verbose=False
    )
    boxes = results[0].boxes
    count = 0
    for box in boxes:
        cls = int(box.cls[0])
        if cls == 0:
            count += 1
    return count


# -------------------------------------------------
# 0) Sağlık kontrolü (test endpoint)
# -------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE}


# -------------------------------------------------
# 1) FOTOĞRAF YÜKLE -> KİŞİ SAYISI
# -------------------------------------------------
@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Tek bir image dosyası alır, kaç kişi olduğunu döner.
    """
    # Dosyayı oku
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Görsel okunamadı"}

    count = count_people_in_frame(img, conf=0.5)

    return {
        "filename": file.filename,
        "person_count": int(count)
    }


# -------------------------------------------------
# 2) VİDEO YÜKLE -> ZAMAN SERİSİ + İSTATİSTİK
# -------------------------------------------------
@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Video alır, saniyeye göre kişi sayısı çıkarır.
    Basitçe her saniyenin ortalama kişi sayısını döner.
    """
    # Gelen videoyu geçici dosyaya kaydet
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_video_path = tmp.name
        contents = await file.read()
        tmp.write(contents)

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return {"error": "Video açılamadı"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # güvenlik için varsayalım

    frame_index = 0
    per_second_counts: Dict[int, List[int]] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        second = int(frame_index / fps)
        frame_index += 1

        # Her saniyeden 1-2 frame bakmak istersen burada "skip" ekleyebiliriz.
        # Şimdilik tüm frameleri sayalım:
        count = count_people_in_frame(frame, conf=0.5)

        if second not in per_second_counts:
            per_second_counts[second] = []
        per_second_counts[second].append(count)

    cap.release()
    os.remove(temp_video_path)  # geçici dosyayı sil

    # Saniye başına ortalama hesapla
    time_series = []
    all_counts = []

    for sec, counts in sorted(per_second_counts.items()):
        avg_c = float(sum(counts) / len(counts))
        time_series.append({"second": sec, "avg_count": avg_c})
        all_counts.extend(counts)

    if all_counts:
        overall_avg = float(sum(all_counts) / len(all_counts))
        max_count = int(max(all_counts))
    else:
        overall_avg = 0.0
        max_count = 0

    return {
        "fps": fps,
        "duration_seconds": max(per_second_counts.keys(), default=0) + 1 if per_second_counts else 0,
        "overall_avg_count": overall_avg,
        "max_count": max_count,
        "time_series": time_series,
        "total_frames": frame_index
    }


# -------------------------------------------------
# 3) CANLI KAMERA İÇİN FRAME ENDPOINT (opsiyonel)
# React tarafında webcam'den frame alıp buraya POST edebilirsin.
# -------------------------------------------------
@app.post("/analyze/frame")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Canlı kamera için: tek kare gönder, kişi sayısını dön.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Görsel okunamadı"}

    count = count_people_in_frame(img, conf=0.5)

    return {
        "person_count": int(count)
    }
