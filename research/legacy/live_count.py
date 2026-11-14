from ultralytics import YOLO
import cv2
import time
import torch

# -----------------------------
# GPU / CPU kontrolü
# -----------------------------
if torch.cuda.is_available():
    device = "cuda"
    print("GPU Kullanılıyor:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("CPU Kullanılıyor")

# -----------------------------
# Model (GPU'ya taşındı)
# -----------------------------
model = YOLO("person_model.pt").to(device)

# Kamera başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

prev_time = time.time()

# Küçük saçma kutuları engellemek için minimum alan
MIN_AREA = 25 * 25

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Çözünürlük (daha doğru için 640x480)
    frame = cv2.resize(frame, (640, 480))

    # FPS hesapla
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    # YOLO tahmini - optimize ayarlar
    results = model(
        frame,
        imgsz=640,
        conf=0.50,     # daha doğru
        iou=0.35,      # yan yana kişiler birleşmez
        classes=[0],   # person
        verbose=False
    )

    detections = results[0].boxes
    person_count = 0

    for box in detections:
        cls = int(box.cls[0])
        if cls == 0:  # person

            # kutu alanı hesabı (gürültüyü azaltır)
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)

            if area < MIN_AREA:
                continue

            person_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Bilgileri ekrana yaz
    cv2.putText(frame, f"People: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Real-Time People Counting (Improved)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
