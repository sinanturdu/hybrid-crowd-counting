import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from model import CSRNet

# ======================
# CONFIG
# ======================
IMAGE_PATH = "./Test/img_0001.jpg"      # 🔴 kendi test görselin
MODEL_PATH = "csrnet_ucf_qnrf_best.pth"

PATCH_SIZE = 512
STRIDE = 256        # overlap (512-256 = %50 overlap)
DOWNSCALE = 8       # dataset ile aynı olmalı

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Kullanılan cihaz:", device)

# ======================
# LOAD MODEL
# ======================
model = CSRNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ======================
# LOAD IMAGE
# ======================
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Görüntü bulunamadı: {IMAGE_PATH}")

img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img = img_rgb.astype(np.float32) / 255.0

H, W, _ = img.shape

# ======================
# PAD IMAGE
# ======================
pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE

img = np.pad(
    img,
    ((0, pad_h), (0, pad_w), (0, 0)),
    mode="constant"
)

H_pad, W_pad, _ = img.shape

# ======================
# FULL DENSITY MAP
# ======================
full_density = np.zeros(
    (H_pad // DOWNSCALE, W_pad // DOWNSCALE),
    dtype=np.float32
)

count_map = np.zeros_like(full_density)

# ======================
# PATCH LOOP
# ======================
with torch.no_grad():
    for y in range(0, H_pad - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W_pad - PATCH_SIZE + 1, STRIDE):

            patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patch_t = (
                torch.from_numpy(patch)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            density = model(patch_t).squeeze().cpu().numpy()

            y_ds = y // DOWNSCALE
            x_ds = x // DOWNSCALE

            full_density[
                y_ds:y_ds + density.shape[0],
                x_ds:x_ds + density.shape[1]
            ] += density

            count_map[
                y_ds:y_ds + density.shape[0],
                x_ds:x_ds + density.shape[1]
            ] += 1

# ======================
# AVERAGE OVERLAPS
# ======================
count_map[count_map == 0] = 1
full_density /= count_map

# ======================
# FINAL COUNT
# ======================
pred_count = full_density.sum()

print(f"\n👥 Tahmini Kişi Sayısı: {pred_count:.2f}")

# ======================
# VISUALIZATION
# ======================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Orijinal Görüntü")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(full_density, cmap="jet")
plt.title(f"Density Map\nTahmini: {pred_count:.1f}")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
