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
img = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ======================
# GT LOAD
# ======================
mat = sio.loadmat(MAT_PATH)

if "annPoints" not in mat:
    raise KeyError("MAT dosyasında 'annPoints' yok")

points = mat["annPoints"]  # (N, 2)
gt_count = points.shape[0]

print(f"👥 Ground Truth Kişi Sayısı: {gt_count}")

# ======================
# VISUALIZATION
# ======================
plt.figure(figsize=(14, 8))

plt.imshow(img_rgb)

# 🔴 Noktaları çiz
plt.scatter(
    points[:, 0],
    points[:, 1],
    s=10,                # nokta boyutu
    c="red",
    marker="x",          # x, o, .
    linewidths=1
)

plt.title(f"GT Noktasal Labellama (Kişi Sayısı: {gt_count})")
plt.axis("off")

plt.tight_layout()
plt.show()
