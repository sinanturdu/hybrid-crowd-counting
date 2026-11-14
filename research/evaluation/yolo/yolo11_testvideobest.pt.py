from ultralytics import YOLO
import cv2
import torch
import os
import subprocess
import sys

# -----------------------------------
# GPU / CPU kontrolü
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Kullanılan cihaz:", device)

# -----------------------------------
# Model ve Video yolu (AYNEN)
# -----------------------------------
model_path = os.getenv("YOLO_MODEL_PATH", "../../backend/models/best.pt")
video_path = os.getenv("VIDEO_PATH", "examples/sample_video.mp4")

# -----------------------------------
# Modeli yükle
# -----------------------------------
model = YOLO(model_path).to(device)

# -----------------------------------
# Video aç
# -----------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Video açılamadı!")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# -----------------------------------
# Çıktı video
# -----------------------------------
os.makedirs("outputs", exist_ok=True)
output_path = os.path.abspath("outputs/test28_detected.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -----------------------------------
# Frame frame analiz
# -----------------------------------
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.4,
        iou=0.5,
        device=device,
        verbose=False
    )

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            conf_score = float(box.conf[0])

            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Person {conf_score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

    out.write(frame)

    if frame_count % 50 == 0:
        print(f"{frame_count}. frame işlendi")

# -----------------------------------
# Kaynakları kapat
# -----------------------------------
cap.release()
out.release()

print("✅ İşlem tamamlandı")
print("📁 Çıktı:", output_path)

# -----------------------------------
# 🔥 VİDEOYU OTOMATİK AÇ
# -----------------------------------
if sys.platform.startswith("win"):
    os.startfile(output_path)
else:
    subprocess.run(["open", output_path])
