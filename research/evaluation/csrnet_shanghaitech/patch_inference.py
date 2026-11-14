import os
import glob
import h5py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from models.csrnet import CSRNet


# ======================================================
# CONFIG
# ======================================================
PATCH_SIZE = 384
STRIDE = 192        # %50 overlap

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ======================================================
# GAUSSIAN WEIGHT (OVERLAP SMOOTHING)
# ======================================================
def gaussian_weight(h, w, sigma=0.5):
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    d = np.sqrt(xx**2 + yy**2)
    g = np.exp(-(d**2) / (2 * sigma**2))
    return torch.from_numpy(g.astype(np.float32))


# ======================================================
# PATCH INFERENCE (CSRNET-CORRECT)
# ======================================================
def patch_inference(model, image, device):
    model.eval()

    img_w, img_h = image.size
    img_tensor = to_tensor(image).to(device)

    H = img_h // 8
    W = img_w // 8

    density_map = torch.zeros((H, W), device=device)
    weight_map  = torch.zeros((H, W), device=device)

    total_count = 0.0  # 🔥 KİŞİ SAYISI BURADAN ALINACAK

    for top in range(0, img_h - PATCH_SIZE + 1, STRIDE):
        for left in range(0, img_w - PATCH_SIZE + 1, STRIDE):

            patch = img_tensor[
                :,
                top:top + PATCH_SIZE,
                left:left + PATCH_SIZE
            ].unsqueeze(0)

            with torch.no_grad():
                pred = model(patch).squeeze(0).squeeze(0)

            # 🔥 PATCH COUNT (KRİTİK)
            total_count += pred.sum().item()

            oh, ow = pred.shape
            y0 = top // 8
            x0 = left // 8

            weight = gaussian_weight(oh, ow).to(device)

            density_map[y0:y0+oh, x0:x0+ow] += pred * weight
            weight_map[y0:y0+oh, x0:x0+ow]  += weight

    # Smooth density map (visual correctness)
    density_map = density_map / (weight_map + 1e-6)

    return total_count


# ======================================================
# MAIN
# ======================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Device:", device)

    WEIGHTS = os.path.join("weights", "csrnet_partA_best.pth")
    img_dir = os.path.join("part_A", "test_data", "images")
    h5_dir  = os.path.join("part_A", "test_data", "ground-truth-h5")

    img_path = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[0]
    name = os.path.splitext(os.path.basename(img_path))[0]
    h5_path = os.path.join(h5_dir, f"{name}.h5")

    image = Image.open(img_path).convert("RGB")

    with h5py.File(h5_path, "r") as f:
        gt_density = np.array(f["density"], dtype=np.float32)
    gt_count = float(gt_density.sum())

    model = CSRNet(load_pretrained_vgg=False).to(device)
    ckpt = torch.load(WEIGHTS, map_location=device)
    model.load_state_dict(ckpt["model"])

    pred_count = patch_inference(model, image, device)

    print("\n==============================")
    print(f"👥 Ground Truth (GT): {gt_count:.0f}")
    print(f"🤖 Prediction (Patch, Gaussian): {pred_count:.2f}")
    print(f"📉 Error: {pred_count - gt_count:.2f}")
    print("==============================\n")


if __name__ == "__main__":
    main()
