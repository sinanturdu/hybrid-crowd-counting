import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from model import CSRNet

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
# PARAMS (EĞİTİMLE AYNI!)
# ======================
PATCH_SIZE = 512
DOWNSCALE = 8

# ======================
# IMAGE PATH
# ======================
IMAGE_PATH = "./Test/img_0001.jpg"   # 🔥 senin yapına uygun

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Görüntü bulunamadı: {IMAGE_PATH}")

# ======================
# LOAD IMAGE
# ======================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("Görüntü okunamadı")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_norm = img_rgb.astype(np.float32) / 255.0

H, W, _ = img_norm.shape

# ======================
# PATCH-BASED INFERENCE
# ======================
total_count = 0.0

density_full = np.zeros((H // DOWNSCALE, W // DOWNSCALE), dtype=np.float32)

with torch.no_grad():
    for y in range(0, H, PATCH_SIZE):
        for x in range(0, W, PATCH_SIZE):

            patch = img_norm[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
                pad_h = PATCH_SIZE - patch.shape[0]
                pad_w = PATCH_SIZE - patch.shape[1]
                patch = np.pad(
                    patch,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant"
                )

            patch_tensor = (
                torch.from_numpy(patch)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            density_patch = model(patch_tensor)
            density_patch = density_patch.squeeze().cpu().numpy()

            # patch density -> full density map'e yerleştir
            dh = density_patch.shape[0]
            dw = density_patch.shape[1]

            y_ds = y // DOWNSCALE
            x_ds = x // DOWNSCALE

            density_full[
                y_ds:y_ds + dh,
                x_ds:x_ds + dw
            ] += density_patch

            total_count += density_patch.sum()

print(f"\n👥 Tahmini Toplam Kişi Sayısı: {total_count:.1f}")

# ======================
# VISUALIZATION
# ======================
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Orijinal Görüntü")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(density_full, cmap="jet")
plt.title(f"Density Map\nTahmini Kişi Sayısı: {total_count:.1f}")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
