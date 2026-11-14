from ultralytics import YOLO
import torch
import os

print("CUDA mevcut mu?:", torch.cuda.is_available())

# --------------------------------------------------
# 1) MODEL OLUŞTUR
# --------------------------------------------------
model = YOLO("yolov8n.pt")   # istersen yolov8s.pt yapabiliriz

# --------------------------------------------------
# 2) DATASET YOLU
# --------------------------------------------------
DATASET_YAML = "data/person_dataset/data.yaml"

if not os.path.exists(DATASET_YAML):
    raise FileNotFoundError(f"YAML bulunamadı! Yol: {DATASET_YAML}")

# --------------------------------------------------
# 3) EĞİTİM
# --------------------------------------------------
model.train(
    data=DATASET_YAML,
    epochs=50,        # istersen 100 yapabiliriz
    imgsz=640,
    batch=16,
    patience=10,     # erken durdurma
    device=0 if torch.cuda.is_available() else "cpu",
)

print("\n🎉 Eğitim tamamlandı!")
print("📁 En iyi model burada olacak: runs/detect/train/weights/best.pt")
