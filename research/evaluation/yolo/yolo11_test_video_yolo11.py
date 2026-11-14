import os
from ultralytics import YOLO
import cv2

# MODEL YÜKLE
model = YOLO(os.getenv("YOLO11_MODEL_PATH", "yolo11/best.pt"))

# TEST EDİLECEK VİDEO
video_path = os.getenv("VIDEO_PATH", "examples/sample_video.mp4")

# ÇIKTILARIN KAYDEDİLECEĞİ KLASÖR
save_dir = os.getenv("OUTPUT_DIR", "outputs/yolo11")

# TAHMİN
results = model.predict(source=video_path, imgsz=640, save=True, project=save_dir, name="result")

print("Bitti! Çıktılar output/result klasörüne kaydedildi.")
