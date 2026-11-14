import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.csrnet import CSRNet
from datasets.shanghaitech import ShanghaiTech


# ======================================================
# CONFIG
# ======================================================
DATA_ROOT = os.getenv("SHANGHAITECH_ROOT", "datasets/ShanghaiTech")
MODEL_PATH = os.getenv("CSRNET_MODEL_PATH", "../../backend/models/csrnet_AB_best.pth")

CROP = 384
SAVE_DIR = "csrnet_eval"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Device:", DEVICE)


# ======================================================
# SUM-PRESERVING RESIZE
# ======================================================
def resize_density_preserve_sum(dens, out_hw):
    old_sum = dens.sum(dim=(2, 3), keepdim=True)
    dens_ds = F.interpolate(dens, size=out_hw, mode="area")
    new_sum = dens_ds.sum(dim=(2, 3), keepdim=True)
    dens_ds = dens_ds * (old_sum / (new_sum + 1e-8))
    return dens_ds


# ======================================================
# SHARPEN
# ======================================================
def sharpen_density(dens, alpha=0.30):
    blurred = F.avg_pool2d(dens, 3, 1, 1)
    sharp = dens + alpha * (dens - blurred)
    return torch.clamp(sharp, min=0.0)


# ======================================================
# LOAD MODEL
# ======================================================
model = CSRNet(load_pretrained_vgg=False).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

print("✅ Model loaded:", MODEL_PATH)


# ======================================================
# DATASET (TEST A + B)
# ======================================================
test_A = ShanghaiTech(DATA_ROOT, part="A", split="test",
                      train_crop_size=CROP, train_random_crop=False)
test_B = ShanghaiTech(DATA_ROOT, part="B", split="test",
                      train_crop_size=CROP, train_random_crop=False)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([test_A, test_B]),
    batch_size=1, shuffle=False
)


# ======================================================
# EVALUATION
# ======================================================
errors = []
gt_counts = []
pr_counts = []

with torch.no_grad():
    for imgs, dens, _ in tqdm(test_loader, desc="Evaluating"):
        imgs = imgs.to(DEVICE)
        dens = dens.to(DEVICE)

        preds = torch.clamp(F.relu(model(imgs)), max=5.0)
        gt_ds = resize_density_preserve_sum(dens, preds.shape[-2:])
        gt_ds = sharpen_density(gt_ds)

        gt_c = gt_ds.sum().item()
        pr_c = preds.sum().item()

        gt_counts.append(gt_c)
        pr_counts.append(pr_c)
        errors.append(pr_c - gt_c)

errors = np.array(errors)
gt_counts = np.array(gt_counts)
pr_counts = np.array(pr_counts)

# ======================================================
# METRICS
# ======================================================
MAE = np.mean(np.abs(errors))
RMSE = np.sqrt(np.mean(errors ** 2))

under_ratio = np.mean(pr_counts < gt_counts)
over_ratio  = np.mean(pr_counts > gt_counts)

# Recall-like metric (%10 tolerance)
recall_like = np.mean(np.abs(errors) / (gt_counts + 1e-6) < 0.10)

print("\n📊 CSRNet Evaluation Results")
print(f"MAE  : {MAE:.2f}")
print(f"RMSE : {RMSE:.2f}")
print(f"Under-count ratio : {under_ratio:.3f}")
print(f"Over-count ratio  : {over_ratio:.3f}")
print(f"Recall-like (<10% error): {recall_like:.3f}")


# ======================================================
# SAVE HISTOGRAM
# ======================================================
plt.figure()
plt.hist(errors, bins=50)
plt.xlabel("Predicted Count - GT Count")
plt.ylabel("Frequency")
plt.title("Count Error Distribution")
plt.savefig(os.path.join(SAVE_DIR, "count_error_hist.png"))
plt.close()


# ======================================================
# SAVE SUMMARY TEXT
# ======================================================
with open(os.path.join(SAVE_DIR, "summary.txt"), "w") as f:
    f.write(f"MAE: {MAE:.2f}\n")
    f.write(f"RMSE: {RMSE:.2f}\n")
    f.write(f"Under-count ratio: {under_ratio:.3f}\n")
    f.write(f"Over-count ratio: {over_ratio:.3f}\n")
    f.write(f"Recall-like (<10% error): {recall_like:.3f}\n")

print(f"\n📁 Results saved to: {SAVE_DIR}")
