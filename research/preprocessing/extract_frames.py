import cv2
import os

VIDEO_PATH = "test4.mp4"   # hangi videodan frame alınacak
OUTPUT_DIR = "custom_frames"
FRAME_COUNT = 50           # çıkarılacak kare sayısı

# Klasör oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Her kaç frame'de bir kare alacağımızı hesaplayalım
step = max(1, total_frames // FRAME_COUNT)

print(f"Toplam frame: {total_frames}, Her {step}. frame kaydedilecek...")

idx = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if idx % step == 0 and saved < FRAME_COUNT:
        save_path = os.path.join(OUTPUT_DIR, f"frame_{saved:03d}.jpg")
        cv2.imwrite(save_path, frame)
        saved += 1

    idx += 1

cap.release()
print(f"{saved} adet frame başarıyla çıkarıldı!")
