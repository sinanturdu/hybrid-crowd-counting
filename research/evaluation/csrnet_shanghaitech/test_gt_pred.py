import os
import glob
import h5py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from models.csrnet import CSRNet


# ======================================================
# TRANSFORM
# ======================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ======================================================
# MAIN
# ======================================================
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Device:", DEVICE)

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------
    WEIGHTS = os.path.join("weights", "csrnet_partA_best.pth")

    # --------------------------------------------------
    # IMAGE (AUTO)
    # --------------------------------------------------
    img_dir = os.path.join("part_A", "test_data", "images")
    h5_dir  = os.path.join("part_A", "test_data", "ground-truth-h5")

    img_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if len(img_list) == 0:
        raise RuntimeError("❌ Test image bulunamadı")

    img_path = img_list[0]  # ilk test görseli
    name = os.path.splitext(os.path.basename(img_path))[0]
    h5_path = os.path.join(h5_dir, f"{name}.h5")

    print("📷 Image:", img_path)

    # --------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------
    img = Image.open(img_path).convert("RGB")

    # --------------------------------------------------
    # LOAD GT
    # --------------------------------------------------
    if not os.path.exists(h5_path):
        raise RuntimeError(f"❌ GT bulunamadı: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        gt_density = np.array(f["density"], dtype=np.float32)

    gt_count = float(gt_density.sum())

    # --------------------------------------------------
    # MODEL LOAD
    # --------------------------------------------------
    model = CSRNet(load_pretrained_vgg=False).to(DEVICE)
    ckpt = torch.load(WEIGHTS, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------
    x = to_tensor(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_density = model(x)
        pred_count = float(pred_density.sum().item())

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------
    print("\n==============================")
    print(f"👥 Ground Truth (GT): {gt_count:.0f}")
    print(f"🤖 Prediction (Pred): {pred_count:.2f}")
    print(f"📉 Error (Pred - GT): {pred_count - gt_count:.2f}")
    print("==============================\n")

    # --------------------------------------------------
    # SHOW IMAGE
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(
        f"{name}\nGT: {int(gt_count)} | Pred: {pred_count:.1f}",
        fontsize=12
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
