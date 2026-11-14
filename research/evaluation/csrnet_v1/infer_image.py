import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ucf_qnrf.model import CSRNet

# torchvision normalize için
from torchvision import transforms

# ======================
# ARGS
# ======================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--image",
    type=str,
    default="../datasets/ShanghaiTech/test/images/part_A_IMG_1.jpg",
    help="Test edilecek görsel yolu"
)
args = parser.parse_args()

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Kullanılan cihaz:", device)

# ======================
# MODEL LOAD (güvenli)
# ======================
model = CSRNet()

def safe_load_weights(path, device):
    # torch 2.1+ ile weights_only denenebilir
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Eski torch sürümleri weights_only desteklemez
        state = torch.load(path, map_location=device)
    return state

WEIGHT_PATH = "csrnet_shanghaitech.pth"
if not os.path.exists(WEIGHT_PATH):
    raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {WEIGHT_PATH}")

model.load_state_dict(safe_load_weights(WEIGHT_PATH, device))
model.to(device)
model.eval()

# ======================
# IMAGE PATH
# ======================
IMAGE_PATH = args.image

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Görüntü bulunamadı: {IMAGE_PATH}")

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise RuntimeError(f"Görüntü okunamadı: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ======================
# PREPROCESS (🔥 kritik kısım)
# CSRNet genelde VGG backbone kullandığı için ImageNet normalize şart.
# ======================
to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),  # [0,1] + CHW
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img_tensor = to_tensor_norm(img_rgb).unsqueeze(0).to(device)

# ======================
# INFERENCE
# ======================
with torch.no_grad():
    density_map = model(img_tensor)

# Negatif değerleri engelle (count negatif çıkmasın)
density_map = density_map.clamp(min=0)

density_np = density_map.squeeze().detach().cpu().numpy()
pred_count = float(density_np.sum())

print(f"👥 Tahmini Kişi Sayısı: {pred_count:.2f}")

# ======================
# VISUALIZATION
# ======================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Orijinal Görüntü")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(density_np, cmap="jet")
plt.title(f"Density Map\nTahmini Kişi Sayısı: {pred_count:.1f}")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
