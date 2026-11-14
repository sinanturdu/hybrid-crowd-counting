import os
import glob
import time
import numpy as np
import h5py
import scipy
import scipy.io
import scipy.ndimage
from PIL import Image
from joblib import Parallel, delayed

# =========================
# AYARLAR
# =========================
DATASET_ROOT = os.getenv("SHANGHAITECH_ROOT", "datasets/ShanghaiTech")  # burayı kontrol et
N_JOBS = 4  # CPU thread sayısı (çok yükseltme, RAM şişer)

# =========================
# DENSITY MAP ÜRETİMİ
# =========================
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))  # (x,y)
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
    distances, _ = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0

        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 4.0

        density += scipy.ndimage.gaussian_filter(pt2d, sigma, mode="constant")

    return density

def generate_density_map(img_path):
    # IMG_x.jpg -> GT_IMG_x.mat
    mat_path = img_path.replace(".jpg", ".mat") \
                      .replace("images", "ground-truth") \
                      .replace("IMG_", "GT_IMG_")

    if not os.path.exists(mat_path):
        print("❌ MAT yok:", mat_path)
        return

    # image size
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # read points
    mat = scipy.io.loadmat(mat_path)
    pts = mat["image_info"][0, 0][0, 0][0]  # Nx2 (x,y)

    k = np.zeros((h, w), dtype=np.float32)
    for x, y in pts:
        x = int(x)
        y = int(y)
        if 0 <= x < w and 0 <= y < h:
            k[y, x] = 1.0

    density = gaussian_filter_density(k)

    # output: images/IMG_x.jpg -> ground-truth-h5/IMG_x.h5
    out_path = img_path.replace("images", "ground-truth-h5").replace(".jpg", ".h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, "w") as hf:
        hf["density"] = density

    return img_path

def collect_image_paths(root):
    sets = []
    for part in ["part_A", "part_B"]:
        for split in ["train_data", "test_data"]:
            sets.append(os.path.join(root, part, split, "images"))

    all_imgs = []
    for folder in sets:
        all_imgs += glob.glob(os.path.join(folder, "*.jpg"))
    return all_imgs

if __name__ == "__main__":
    start = time.time()
    imgs = collect_image_paths(DATASET_ROOT)
    print("Toplam image:", len(imgs))

    Parallel(n_jobs=N_JOBS)(delayed(generate_density_map)(p) for p in imgs)

    print("✅ Bitti. Süre:", time.time() - start, "sn")
