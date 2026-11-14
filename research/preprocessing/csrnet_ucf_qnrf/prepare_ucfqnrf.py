import os
import cv2
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from tqdm import tqdm

ROOT = os.getenv("UCF_QNRF_ROOT", "datasets/ucf_qnrf")
DOWNSAMPLE = 8   # CSRNet standard

def load_points(mat_path):
    mat = loadmat(mat_path)
    if "annPoints" not in mat:
        raise ValueError(f"annPoints yok: {mat_path}")
    return mat["annPoints"].astype(np.float32)

def make_density(h, w, points):
    h_d = h // DOWNSAMPLE
    w_d = w // DOWNSAMPLE
    density = np.zeros((h_d, w_d), dtype=np.float32)

    if len(points) == 0:
        return density

    pts = points.copy()
    pts[:, 0] /= DOWNSAMPLE
    pts[:, 1] /= DOWNSAMPLE

    pts[:, 0] = np.clip(pts[:, 0], 0, w_d - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h_d - 1)

    tree = KDTree(pts)

    for i, (x, y) in enumerate(pts):
        dists, _ = tree.query([x, y], k=min(4, len(pts)))
        sigma = max(4.0, 0.3 * np.mean(dists[1:])) if len(pts) > 1 else 8.0

        ix, iy = int(round(x)), int(round(y))
        density[iy, ix] += 1
        density = gaussian_filter(density, sigma=sigma)

    density *= len(points) / max(density.sum(), 1e-6)
    return density

def process(split):
    img_dir = os.path.join(ROOT, split)
    out_dir = os.path.join(ROOT, "density", split)
    os.makedirs(out_dir, exist_ok=True)

    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

    for img_name in tqdm(imgs, desc=f"{split} density"):
        base = img_name.replace(".jpg", "")
        img_path = os.path.join(img_dir, img_name)
        mat_path = os.path.join(img_dir, base + "_ann.mat")

        img = cv2.imread(img_path)
        if img is None or not os.path.exists(mat_path):
            continue

        h, w = img.shape[:2]
        points = load_points(mat_path)

        density = make_density(h, w, points)
        np.save(os.path.join(out_dir, base + ".npy"), density)

if __name__ == "__main__":
    process("Train")
    process("Test")
    print("✅ Density map’ler hazır (aynı klasör yapısına göre)")
