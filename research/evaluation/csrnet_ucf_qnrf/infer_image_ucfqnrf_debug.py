import torch
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from model import CSRNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Kullanılan cihaz:", DEVICE)

PATCH = 512
STRIDE = 512
DOWNSCALE = 8

model = CSRNet().to(DEVICE)
model.load_state_dict(torch.load("csrnet_ucf_qnrf_best.pth", map_location=DEVICE))
model.eval()

img_path = "Test/img_0001.jpg"
mat_path = "Test/img_0001_ann.mat"

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_norm = img_rgb.astype(np.float32) / 255.0

H, W, _ = img_norm.shape
full_density = np.zeros((H // DOWNSCALE, W // DOWNSCALE), dtype=np.float32)

with torch.no_grad():
    for y in range(0, H - PATCH + 1, STRIDE):
        for x in range(0, W - PATCH + 1, STRIDE):
            patch = img_norm[y:y+PATCH, x:x+PATCH]
            patch_t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            pred = model(patch_t).squeeze().cpu().numpy()

            yy = y // DOWNSCALE
            xx = x // DOWNSCALE
            full_density[yy:yy+pred.shape[0], xx:xx+pred.shape[1]] += pred

pred_count = full_density.sum()

mat = sio.loadmat(mat_path)
gt_points = mat["annPoints"]
gt_count = gt_points.shape[0]

print(f"👥 GT Count   : {gt_count}")
print(f"🤖 Pred Count : {pred_count:.1f}")
print(f"📉 Diff       : {abs(gt_count - pred_count):.1f}")

plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title("Orijinal Görüntü")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(full_density, cmap="jet")
plt.title("Predicted Density (Patch-based)")
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(img_rgb)
plt.scatter(gt_points[:,0], gt_points[:,1], s=2, c="lime")
plt.title(f"GT Noktaları ({gt_count})")
plt.axis("off")

plt.show()
