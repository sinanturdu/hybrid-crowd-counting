from ultralytics import YOLO
import cv2
import pandas as pd
import time
import os
import torch

# ---------------------------
# GPU / CPU
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Kullanılan cihaz:", device)

# ---------------------------
# MODEL
# ---------------------------
model = YOLO("person_model.pt").to(device)

# ---------------------------
# VİDEO
# ---------------------------
VIDEO_NAME = "test28.mp4"

if not os.path.exists(VIDEO_NAME):
    raise FileNotFoundError(f"Video bulunamadı: {VIDEO_NAME}")

cap = cv2.VideoCapture(VIDEO_NAME)
if not cap.isOpened():
    raise RuntimeError(f"Video açılamadı: {VIDEO_NAME}")

results_list = []
frame_idx = 0
start = time.time()

# Daha küçük insanları da yakalaması için düşürüldü
MIN_AREA = 20 * 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    H, W, _ = frame.shape

    results = model.predict(
        frame,
        imgsz=W,
        conf=0.35,    # artırılmış duyarlılık
        iou=0.45,     # daha stabil NMS
        classes=[0],
        verbose=False
    )

    boxes = results[0].boxes
    person_count = 0

    for box in boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        w = float(x2 - x1)
        h = float(y2 - y1)
        area = w * h

        if area < MIN_AREA:
            continue

        person_count += 1

        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 2)

    timestamp = round(time.time() - start, 2)

    results_list.append({
        "frame": frame_idx,
        "timestamp": timestamp,
        "count": person_count
    })

    cv2.putText(frame, f"Count: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("People Count - Universal Optimized", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(results_list)
df.to_csv("counts.csv", index=False)

print("counts.csv başarıyla oluşturuldu!")
print("Toplam tespit edilen kare:", len(results_list))
