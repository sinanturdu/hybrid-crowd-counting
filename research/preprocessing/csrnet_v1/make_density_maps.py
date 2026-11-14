import os
import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter

ROOT = "../datasets/ShanghaiTech"
SIGMA = 15


def extract_points(mat):
    try:
        pts = mat["image_info"][0][0][0][0]

        # tuple → numpy
        if isinstance(pts, tuple):
            pts = np.array(pts)

        if pts is None or len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        pts = np.array(pts)

        # (2,) → (1,2)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        # (N,1,2) vb. → (N,2)
        if pts.ndim > 2:
            pts = pts.reshape(pts.shape[0], -1)

        return pts.astype(np.float32)

    except Exception:
        return np.zeros((0, 2), dtype=np.float32)


def generate_density(image, points):
    h, w, _ = image.shape
    density = np.zeros((h, w), dtype=np.float32)

    # ❌ shape kullanma, tuple patlatır
    if points is None or len(points) == 0:
        return density

    for i in range(len(points)):
        try:
            x = int(points[i][0])
            y = int(points[i][1])
        except:
            continue

        if 0 <= x < w and 0 <= y < h:
            density[y, x] += 1

    density = gaussian_filter(density, sigma=SIGMA)
    return density


def process_split(split):
    img_dir = os.path.join(ROOT, split, "images")
    gt_dir = os.path.join(ROOT, split, "ground_truth")

    for fname in os.listdir(img_dir):
        img_path = os.path.join(img_dir, fname)
        gt_name = fname.replace("IMG", "GT_IMG").replace(".jpg", ".mat")
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(gt_path):
            continue

        img = cv2.imread(img_path)
        mat = sio.loadmat(gt_path)

        points = extract_points(mat)
        density = generate_density(img, points)

        sio.savemat(gt_path, {"density": density})

    print(f"✅ {split} density map üretildi")


process_split("train")
process_split("test")
print("🎉 Density map üretimi TAMAMLANDI!")
