import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# ======================
# PATH AYARLARI
# ======================
IMG_PATH = "Train/img_0001.jpg"        # değiştir
MAT_PATH = "Train/img_0001_ann.mat"    # değiştir

if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Görüntü yok: {IMG_PATH}")

if not os.path.exists(MAT_PATH):
    raise FileNotFoundError(f"MAT yok: {MAT_PATH}")

# ======================
# IMAGE LOAD
# ======================
img_bgr = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ======================
# GT LOAD
# ======================
mat = sio.loadmat(MAT_PATH)

if "annPoints" not in mat:
    raise KeyError("MAT dosyasında 'annPoints' yok")

points = mat["annPoints"]
gt_count = points.shape[0]

print(f"👥 GT Kişi Sayısı: {gt_count}")

# ======================
# OVERLAY IMAGE (şeffaf)
# ======================
overlay = img_rgb.copy()

for x, y in points:
    cv2.circle(
        overlay,
        (int(x), int(y)),
        radius=3,
        color=(255, 0, 0),   # kırmızı
        thickness=-1
    )

alpha = 0.6
overlay_img = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)

# ======================
# VISUALIZATION
# ======================
plt.figure(figsize=(20, 7))

# 1️⃣ Orijinal
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Orijinal Görüntü")
plt.axis("off")

# 2️⃣ Yan Yana – Noktalı
plt.subplot(1, 3, 2)
plt.imshow(img_rgb)
plt.scatter(
    points[:, 0],
    points[:, 1],
    s=10,
    c="red",
    marker="x"
)
plt.title(f"GT Noktasal Label (Kişi: {gt_count})")
plt.axis("off")

# 3️⃣ Şeffaf Overlay
plt.subplot(1, 3, 3)
plt.imshow(overlay_img)
plt.title("Overlay (Orijinal + GT Noktaları)")
plt.axis("off")

plt.tight_layout()
plt.show()
