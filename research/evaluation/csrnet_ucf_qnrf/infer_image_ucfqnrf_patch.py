import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from model import CSRNet
from skimage.feature import peak_local_max

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Kullanılan cihaz:", device)

# ======================
# MODEL
# ======================
model = CSRNet().to(device)
model.load_state_dict(torch.load("csrnet_ucf_qnrf_best.pth", map_location=device))
model.eval()

# ======================
# IMAGE PATH
# ======================
IMAGE_PATH = "./Test/img_0001.jpg"

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Görüntü bulunamadı: {IMAGE_PATH}")

# ======================
# LOAD IMAGE
# ======================
img = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_norm = img_rgb.astype(np.float32) / 255.0

H, W, _ = img_rgb.shape

# ======================
# PATCH SETTINGS
# ======================
PATCH = 512
STRIDE = PATCH        # ❌ overlap yok
DOWNSCALE = 8

# 🔥 Global calibration (testten aldığın)
SCALE_FACTOR = 0.79

dh, dw = H // DOWNSCALE, W // DOWNSCALE
density_final = np.zeros((dh, dw), dtype=np.float32)

# ======================
# PATCH INFERENCE
# ======================
with torch.no_grad():
    for y in range(0, H - PATCH + 1, STRIDE):
        for x in range(0, W - PATCH + 1, STRIDE):

            patch = img_norm[y:y+PATCH, x:x+PATCH]

            patch_tensor = (
                torch.from_numpy(patch)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            density_patch = model(patch_tensor)
            density_patch = density_patch.squeeze().cpu().numpy()

            # negatifleri sıfırla
            density_patch = np.maximum(density_patch, 0)

            h_ds = y // DOWNSCALE
            w_ds = x // DOWNSCALE

            density_final[
                h_ds:h_ds + density_patch.shape[0],
                w_ds:w_ds + density_patch.shape[1]
            ] += density_patch

# ======================
# SCALE CALIBRATION
# ======================
density_final *= SCALE_FACTOR
density_count = density_final.sum()

# ==========================================================
# 🔴 DENSITY → NOKTA (PEAK-BASED COUNT)
# ==========================================================
density_for_peaks = density_final.copy()

# gürültüyü bastır
density_for_peaks[density_for_peaks < 0.015] = 0

points = peak_local_max(
    density_for_peaks,
    min_distance=4,       # iki insan arası min mesafe
    threshold_abs=0.02    # eşik
)

point_count = len(points)

# ======================
# VISUALIZATION
# ======================
plt.figure(figsize=(18, 6))

# Orijinal
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Orijinal Görüntü")
plt.axis("off")

# Density
plt.subplot(1, 3, 2)
plt.imshow(density_final, cmap="jet")
plt.title(f"Density Map\nCount (sum): {density_count:.1f}")
plt.colorbar()
plt.axis("off")

# Noktalar
plt.subplot(1, 3, 3)
plt.imshow(img_rgb)
plt.scatter(
    points[:, 1] * DOWNSCALE,
    points[:, 0] * DOWNSCALE,
    s=6,
    c="red"
)
plt.title(f"Peak-Based Points\nCount (points): {point_count}")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"👥 Density-based Count : {density_count:.1f}")
print(f"🔴 Peak-based Count   : {point_count}")
