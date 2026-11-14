from ultralytics import YOLO

# -------------------------------
# MODELİN YÜKLENMESİ (Yolov9c)
# -------------------------------
model = YOLO("yolov9c.pt")  # eğitim başlangıç modeli

# -------------------------------
# EĞİTİM
# -------------------------------
model.train(
    data="data.yaml",
    epochs=200,
    imgsz=1280,
    batch=4,
    workers=2,
    lr0=0.002,
    optimizer="AdamW",
    mosaic=1.0,
    mixup=0.2,
    hsv_h=0.015,
    degrees=5,
    translate=0.1,
    scale=0.8,
    flipud=0.2,
    fliplr=0.5,
    amp=True,
    patience=40,
)

print("🔥 Eğitim tamamlandı! Sonuçlar runs/detect/trainX klasörüne kaydedildi.")
