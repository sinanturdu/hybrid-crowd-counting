import os
import h5py
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from PIL import Image

DATASET_ROOT = "datasets/ShanghaiTech"

def generate_density(image, points, sigma=15):
    h, w = image.size[1], image.size[0]
    density = np.zeros((h, w), dtype=np.float32)

    for point in points:
        x = min(w - 1, max(0, int(point[0])))
        y = min(h - 1, max(0, int(point[1])))
        density[y, x] += 1

    density = gaussian_filter(density, sigma=sigma)
    return density

def process_split(split):
    print(f"\n🔄 {split.upper()} işleniyor...")

    img_dir = os.path.join(DATASET_ROOT, split, "images")
    gt_dir = os.path.join(DATASET_ROOT, split, "ground_truth")
    out_dir = os.path.join(DATASET_ROOT, split, "density_maps")

    os.makedirs(out_dir, exist_ok=True)

    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(
            gt_dir, "GT_" + os.path.splitext(img_name)[0] + ".mat"
        )

        if not os.path.exists(gt_path):
            print(f"⚠️ GT yok: {img_name}")
            continue

        image = Image.open(img_path).convert("RGB")
        mat = sio.loadmat(gt_path)

        points = mat["image_info"][0][0][0][0][0]

        density = generate_density(image, points)

        with h5py.File(
            os.path.join(out_dir, img_name.replace(".jpg", ".h5")), "w"
        ) as hf:
            hf["density"] = density

    print(f"✅ {split.upper()} tamamlandı")

process_split("train")
process_split("test")

print("\n🎉 Density map üretimi tamamlandı!")
