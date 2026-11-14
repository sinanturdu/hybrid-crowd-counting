from ultralytics import YOLO
import cv2
import os

# -------------------------------
# PATHLER
# -------------------------------
MODEL_PATH = r"yolo8\runs\detect\train\weights\best.pt"
IMAGE_PATH = os.getenv("IMAGE_PATH", "examples/sample.jpg")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "outputs/photo_result.jpg")

# -------------------------------
# MODEL YÜKLE
# -------------------------------
model = YOLO(MODEL_PATH)

# -------------------------------
# GÖRÜNTÜ OKU
# -------------------------------
image = cv2.imread(IMAGE_PATH)
assert image is not None, "❌ Görüntü okunamadı!"

# -------------------------------
# TAHMİN
# -------------------------------
results = model(
    image,
    conf=0.20,   # Recall'u biraz artırmak için
    iou=0.6,
    device=0
)

person_count = 0
MIN_AREA = 25 * 25  # Çok küçük sahte tespitleri elemek için

# -------------------------------
# KUTULARI ÇİZ + SAY
# -------------------------------
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        area = (x2 - x1) * (y2 - y1)
        if area < MIN_AREA:
            continue

        person_count += 1

        label = f"person {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

# -------------------------------
# SONUÇLARI YAZDIR
# -------------------------------
cv2.putText(
    image,
    f"Total People: {person_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

cv2.imwrite(OUTPUT_PATH, image)

print("✅ İnsan sayımı tamamlandı")
print(f"👤 Toplam insan sayısı: {person_count}")
print(f"📸 Çıktı görseli: {OUTPUT_PATH}")

# -------------------------------
# OTOMATİK AÇ
# -------------------------------
os.startfile(OUTPUT_PATH)
