from ultralytics import YOLO
import cv2
import os

# -------------------------------
# PATHLER
# -------------------------------
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "../../backend/models/best2.pt")
VIDEO_PATH = os.getenv("VIDEO_PATH", "examples/sample_video.mp4")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "outputs/video_result.mp4")

# -------------------------------
# MODEL YÜKLE
# -------------------------------
model = YOLO(MODEL_PATH)

# -------------------------------
# VIDEO OKU
# -------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "❌ Video açılamadı!"

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# -------------------------------
# FRAME FRAME İŞLE
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(
        frame,
        conf=0.25,   # Sonra 0.15 deneyebiliriz (recall ↑)
        iou=0.6,
        device=0
    )

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            label = f"person {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    out.write(frame)

cap.release()
out.release()

print("✅ Video işlendi ve kaydedildi:")
print(OUTPUT_PATH)

# -------------------------------
# VIDEOYU OTOMATİK AÇ
# -------------------------------
os.startfile(OUTPUT_PATH)
