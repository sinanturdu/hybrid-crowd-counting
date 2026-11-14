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
# CONFIG (CSRNET SAFE)
# ======================================================
PATCH_SIZE = 384
STRIDE = 384   # ❗ NO OVERLAP

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ======================================================
# PATCH INFERENCE
# ======================================================
def patch_inference(model, image, device):
    model.eval()

    img_w, img_h = image.size
    img_tensor = to_tensor(image).to(device)

    density_map = torch.zeros(
        (img_h // 8, img_w // 8),
        device=device
    )

    for top in range(0, img_h, STRIDE):
        for left in range(0, img_w, STRIDE):

            bottom = min(top + PATCH_SIZE, img_h)
            right = min(left + PATCH_SIZE, img_w)

            patch = img_tensor[:, top:bottom, left:right]
            ph, pw = patch.shape[1:]

            if ph < PATCH_SIZE or pw < PATCH_SIZE:
                patch = F.pad(
                    patch,
                    (0, PATCH_SIZE - pw, 0, PATCH_SIZE - ph)
                )

            patch = patch.unsqueeze(0)

            with torch.no_grad():
                pred = model(patch).squeeze(0).squeeze(0)

            oh = ph // 8
            ow = pw // 8
            pred = pred[:oh, :ow]

            y0 = top // 8
            x0 = left // 8

            density_map[y0:y0+oh, x0:x0+ow] += pred

    return float(density_map.sum().item())


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
    print(f"🤖 Prediction (Patch, fixed): {pred_count:.2f}")
    print(f"📉 Error: {pred_count - gt_count:.2f}")
    print("==============================\n")


if __name__ == "__main__":
    main()
