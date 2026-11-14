import os
import sys
from ultralytics import YOLO
import cv2

# Model yolu (kendi eğittiğin YOLO modeli!!)
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/best.pt")

# Python argümanı: resim yolu
image_path = sys.argv[1]

# Model yükle
model = YOLO(MODEL_PATH)

# Resmi oku
img = cv2.imread(image_path)

if img is None:
    print("0")
    sys.exit()

# Tahmin
results = model(img)[0]

# İnsan sayısını bul
person_count = 0
for box in results.boxes:
    if int(box.cls[0]) == 0:
        person_count += 1

# Sadece sayı bas — Node.js bunu okuyacak!
print(person_count)
