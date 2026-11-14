import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import CSRNet

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "csrnet_ucf_qnrf_best.pth"

IMAGE_PATH = "test.jpg"   # ← test edeceğin foto
MAX_SIZE = 1024           # OOM koruması

print("🔥 Kullanılan cihaz:", DEVICE)

# =========================
# MODEL
# =========================
model = CSRNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================
# IMAGE LOAD
# =========================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Görüntü bulunamadı: {IMAGE_PATH}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# =========================
# RESIZE (TRAIN İLE UYUMLU)
# =========================
scale = 1.0
max_dim = max(h, w)
if max_dim > MAX_SIZE:
    scale = MAX_SIZE / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

img = img.astype(np.float32) / 255.0

# =========================
# TENSOR
# =========================
img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# =========================
# INFERENCE
# =========================
with torch.no_grad():
    density = model(img_t)

density = torch.relu(density)      # NEGATİFİ TEMİZLE
count = density.sum().item()

print(f"\n👥 Tahmini Kişi Sayısı: {count:.2f}")

# =========================
# VISUALIZATION
# =========================
density_map = density.squeeze().cpu().numpy()

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(density_map, cmap="jet")
plt.title(f"Density Map (Count={count:.1f})")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
