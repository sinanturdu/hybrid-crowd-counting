import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import CSRNet

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "csrnet_ucf_qnrf_best.pth"
IMAGE_PATH = os.getenv("IMAGE_PATH", "examples/sample.jpg")

PATCH_SIZE = 512
STRIDE = 256
DOWNSCALE = 8

# =========================
# LOAD MODEL
# =========================
model = CSRNet().to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)

# 🔥 ANAHTAR DÜZELTME
new_state = {}
for k, v in state.items():
    if k.startswith("output_layer"):
        new_state[k.replace("output_layer", "output")] = v
    else:
        new_state[k] = v

model.load_state_dict(new_state, strict=True)
model.eval()

print("🔥 Model yüklendi | Device:", DEVICE)

# =========================
# LOAD IMAGE
# =========================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("Görüntü okunamadı!")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

img_f = img.astype(np.float32) / 255.0
img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# =========================
# PAD IMAGE
# =========================
pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE

img_t = torch.nn.functional.pad(
    img_t,
    (0, pad_w, 0, pad_h),
    mode="constant"
)

_, _, H, W = img_t.shape

# =========================
# PATCH INFERENCE
# =========================
density_map = np.zeros((H // DOWNSCALE, W // DOWNSCALE), dtype=np.float32)

with torch.no_grad():
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            patch = img_t[:, :, y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            pred = model(patch)
            pred = pred.squeeze().cpu().numpy()

            dy = y // DOWNSCALE
            dx = x // DOWNSCALE

            ph, pw = pred.shape
            density_map[dy:dy + ph, dx:dx + pw] += pred

# =========================
# COUNT
# =========================
count = density_map.sum()
print(f"\n👥 Tahmini kişi sayısı: {count:.1f}")

# =========================
# VISUALIZE
# =========================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Predicted Density Map | Count = {count:.1f}")
plt.imshow(density_map, cmap="jet")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
