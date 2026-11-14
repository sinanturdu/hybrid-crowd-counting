import os
import h5py
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.csrnet import CSRNet


# ======================================================
# CONFIG
# ======================================================
PATCH_SIZE = 384
STRIDE = 192   # %50 overlap

IMG_NAMES = ["IMG_24.jpg", "IMG_86.jpg"]
PART = "A"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ======================================================
# LOAD IMAGE & GT
# ======================================================
def load_image(img_path):
    return Image.open(img_path).convert("RGB")


def load_gt_density(h5_path):
    with h5py.File(h5_path, "r") as f:
        density = np.array(f["density"], dtype=np.float32)
    return density


# ======================================================
# GAUSSIAN WEIGHT (BLENDING)
# ======================================================
_WEIGHT_CACHE = {}

def gaussian_weight(h, w, sigma=0.5):
    key = (h, w, float(sigma))
    if key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[key]

    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    d = np.sqrt(xx**2 + yy**2)
    g = np.exp(-(d**2) / (2 * sigma**2))
    wt = torch.from_numpy(g.astype(np.float32))
    _WEIGHT_CACHE[key] = wt
    return wt


# ======================================================
# PATCH-BASED INFERENCE
# ======================================================
@torch.no_grad()
def patch_inference(model, image, device):
    model.eval()

    img_w, img_h = image.size
    img_tensor = to_tensor(image).to(device)

    out_h = int(math.ceil(img_h / 8.0))
    out_w = int(math.ceil(img_w / 8.0))

    density_map = torch.zeros((out_h, out_w), device=device)
    norm_map = torch.zeros((out_h, out_w), device=device)

    tops = list(range(0, max(1, img_h - PATCH_SIZE + 1), STRIDE))
    lefts = list(range(0, max(1, img_w - PATCH_SIZE + 1), STRIDE))

    if tops[-1] != max(0, img_h - PATCH_SIZE):
        tops.append(max(0, img_h - PATCH_SIZE))
    if lefts[-1] != max(0, img_w - PATCH_SIZE):
        lefts.append(max(0, img_w - PATCH_SIZE))

    for top in tops:
        for left in lefts:
            patch = img_tensor[:, top:top + PATCH_SIZE, left:left + PATCH_SIZE]

            ph, pw = patch.shape[1], patch.shape[2]
            if ph < PATCH_SIZE or pw < PATCH_SIZE:
                patch = F.pad(patch, (0, PATCH_SIZE - pw, 0, PATCH_SIZE - ph))

            pred = model(patch.unsqueeze(0))
            pred = torch.clamp(F.relu(pred), max=5.0)
            pred = pred.squeeze(0).squeeze(0)

            real_h = min(PATCH_SIZE, img_h - top)
            real_w = min(PATCH_SIZE, img_w - left)

            real_oh = int(math.ceil(real_h / 8.0))
            real_ow = int(math.ceil(real_w / 8.0))

            pred = pred[:real_oh, :real_ow]

            y0, x0 = top // 8, left // 8
            weight = gaussian_weight(*pred.shape).to(device)

            y1 = min(out_h, y0 + pred.shape[0])
            x1 = min(out_w, x0 + pred.shape[1])

            density_map[y0:y1, x0:x1] += pred[:y1-y0, :x1-x0] * weight[:y1-y0, :x1-x0]
            norm_map[y0:y1, x0:x1] += weight[:y1-y0, :x1-x0]

    density_map = density_map / (norm_map + 1e-6)

    pred_low = density_map.cpu().numpy()
    pred_count = int(pred_low.sum())

    pred_vis = F.interpolate(
        density_map.unsqueeze(0).unsqueeze(0),
        size=(img_h, img_w),
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()

    return pred_vis, pred_count


# ======================================================
# MAIN
# ======================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Device:", device)

    DATA_ROOT = os.getenv("SHANGHAITECH_ROOT", "datasets/ShanghaiTech")
    WEIGHTS = os.path.join("weights", "csrnet_AB_best.pth")

    part_dir = "part_A" if PART.upper() == "A" else "part_B"
    img_dir = os.path.join(DATA_ROOT, part_dir, "test_data", "images")
    h5_dir  = os.path.join(DATA_ROOT, part_dir, "test_data", "ground-truth-h5")

    model = CSRNet(load_pretrained_vgg=False).to(device)
    ckpt = torch.load(WEIGHTS, map_location=device)
    model.load_state_dict(ckpt["model"])

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 1, 0.05],
        hspace=0.15,
        wspace=0.1
    )

    for row, img_name in enumerate(IMG_NAMES):
        img_path = os.path.join(img_dir, img_name)
        image = load_image(img_path)

        name = os.path.splitext(img_name)[0]
        gt_density = load_gt_density(os.path.join(h5_dir, f"{name}.h5"))
        gt_count = int(gt_density.sum())

        pred_vis, pred_count = patch_inference(model, image, device)

        img_np = np.array(image)

        ax_orig = fig.add_subplot(gs[row, 0])
        ax_pred = fig.add_subplot(gs[row, 1])
        cax     = fig.add_subplot(gs[row, 2])

        ax_orig.imshow(img_np)
        ax_orig.set_title(f"GT Count = {gt_count}", fontsize=12)
        ax_orig.axis("off")

        im = ax_pred.imshow(pred_vis, cmap="jet")
        ax_pred.set_title(f"Pred Count = {pred_count}", fontsize=12)
        ax_pred.axis("off")

        fig.colorbar(im, cax=cax)

    plt.show()


if __name__ == "__main__":
    main()
