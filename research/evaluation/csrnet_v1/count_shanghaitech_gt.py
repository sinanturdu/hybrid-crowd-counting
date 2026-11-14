import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image

# =====================================================
# AYARLAR
# =====================================================
IMG_ID = 1  # IMG_1.jpg için 1, IMG_10.jpg için 10 vs.

BASE_PATH = "datasets/ShanghaiTech/part_A/train_data"

IMG_PATH = os.path.join(BASE_PATH, "images", f"IMG_{IMG_ID}.jpg")
GT_PATH  = os.path.join(BASE_PATH, "ground-truth", f"GT_IMG_{IMG_ID}.mat")

# =====================================================
# KONTROL
# =====================================================
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Görsel bulunamadı: {IMG_PATH}")

if not os.path.exists(GT_PATH):
    raise FileNotFoundError(f"GT bulunamadı: {GT_PATH}")

# =====================================================
# IMAGE YÜKLE
# =====================================================
img = Image.open(IMG_PATH)
img_np = np.array(img)

# =====================================================
# GT (.mat) YÜKLE
# =====================================================
mat = sio.loadmat(GT_PATH)
points = mat["image_info"][0, 0][0, 0][0]

person_count = points.shape[0]

# =====================================================
# SONUÇ
# =====================================================
print(f"📸 Dosya: IMG_{IMG_ID}.jpg")
print(f"👤 Gerçek kişi sayısı (GT): {person_count}")

# =====================================================
# GÖRSELLEŞTİR
# =====================================================
plt.figure(figsize=(8, 8))
plt.imshow(img_np)
plt.scatter(points[:, 0], points[:, 1], c="red", s=10)
plt.title(f"GT Kişi Sayısı: {person_count}")
plt.axis("off")
plt.show()
