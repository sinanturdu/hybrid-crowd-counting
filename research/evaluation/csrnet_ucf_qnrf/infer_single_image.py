import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import CSRNet

# =========================
# AYARLAR
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Kullanılan cihaz:", DEVICE)

MODEL_PATH = "csrnet_ucf_qnrf_best.pth"
IMG_PATH = os.getenv("IMAGE_PATH", "examples/sample.jpg")

# ⚠️ EĞİTİMDE density downscale ederken *64 yaptıysan bunu 64 yap
# Örn: density = cv2.resize(density, (w//8,h//8)) * 64
COUNT_SCALE = 1  # dene: 1, olmadı 64 dene

# CSRNet genelde /8 çıktı verir; padding için 8'e tamamlayalım
PAD_MULTIPLE = 8

# ImageNet normalize (CSRNet/VGG tabanlı modellerde çoğunlukla şart)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# =========================
# MODEL
# =========================
model = CSRNet().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=True)
model.eval()

# =========================
# IMAGE LOAD + PREPROCESS
# =========================
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise RuntimeError("❌ Görüntü okunamadı")

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# normalize
img = (img - IMAGENET_MEAN) / IMAGENET_STD

h, w, _ = img.shape

# pad to multiple of 8 (modelin downsample katmanları için güvenli)
pad_h = (PAD_MULTIPLE - (h % PAD_MULTIPLE)) % PAD_MULTIPLE
pad_w = (PAD_MULTIPLE - (w % PAD_MULTIPLE)) % PAD_MULTIPLE

img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)

img_t = torch.from_numpy(img_pad).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# =========================
# INFERENCE
# =========================
with torch.no_grad():
    density = model(img_t)
    density = F.relu(density)

# pad’i geri kırp (çıktı ölçeğine göre)
out = density.squeeze(0).squeeze(0).cpu().numpy()

# output boyutu inputtan küçük (genelde /8). Bu yüzden kırpma oranını bulalım:
out_h, out_w = out.shape
# input_pad_h / out_h = scale factor (yaklaşık 8)
# paddingin outputtaki karşılığı:
cut_h = int(round(pad_h * (out_h / (h + pad_h)))) if pad_h > 0 else 0
cut_w = int(round(pad_w * (out_w / (w + pad_w)))) if pad_w > 0 else 0

if cut_h > 0:
    out = out[:-cut_h, :]
if cut_w > 0:
    out = out[:, :-cut_w]

pred_count = float(out.sum() * COUNT_SCALE)
print(f"\n👥 Tahmini kişi sayısı: {pred_count:.2f}")
print(f"🧩 Density map shape: {out.shape}  | COUNT_SCALE={COUNT_SCALE}")

# =========================
# VISUALIZATION
# =========================
# Orijinali gösterelim (normalize edilmiş değil)
img_show = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_show)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Density Map (Count ≈ {pred_count:.1f})")
plt.imshow(out, cmap="jet")
plt.colorbar(fraction=0.046)
plt.axis("off")

plt.tight_layout()
plt.show()
