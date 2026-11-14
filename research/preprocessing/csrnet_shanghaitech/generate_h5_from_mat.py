import os
import glob
import time
import numpy as np
import h5py
import scipy.io
from scipy.spatial import cKDTree
from PIL import Image
from joblib import Parallel, delayed

# =========================
# AYARLAR
# =========================
DATASET_ROOT = os.getenv("SHANGHAITECH_ROOT", "datasets/ShanghaiTech")  # burayı kontrol et
N_JOBS = 4  # CPU thread sayısı (çok yükseltme, RAM şişer)

# =========================
# DENSITY MAP ÜRETİMİ (Clean-Room Implementation)
# Mathematical reference:
#   Zhang et al., "Single-Image Crowd Counting via Multi-Column Convolutional
#   Neural Network", IEEE CVPR 2016, Section 2.1, Eq. (4):
#       sigma_i = beta * d_bar_i,  with beta = 0.3 and m = 3 nearest neighbors.
#   Li et al., "CSRNet: Dilated Convolutional Neural Networks for Understanding
#   the Highly Congested Scenes", IEEE CVPR 2018.
# =========================
def build_geometry_adaptive_density(
    height: int,
    width: int,
    points_xy: np.ndarray,
    beta: float = 0.3,
    k_neighbors: int = 3,
    min_sigma: float = 1.0,
    truncate_stds: float = 4.0,
) -> np.ndarray:
    """
    Clean-room implementation of geometry-adaptive Gaussian density map generation
    from published mathematical equations (Zhang et al., CVPR 2016, Eq. 4).

    Parameters
    ----------
    height, width : int
        Target image dimensions (H, W).
    points_xy : np.ndarray
        Array of (x, y) head annotation coordinates of shape (N, 2).
    beta : float
        Geometry-adaptive scaling factor (0.3 in Zhang et al., CVPR 2016).
    k_neighbors : int
        Number of nearest spatial neighbors used to estimate local crowd spacing.
    min_sigma : float
        Minimum Gaussian standard deviation floor to prevent degenerate zero-width
        kernels when multiple annotations share identical or near-identical coordinates.
    truncate_stds : float
        Local window truncation radius in multiples of sigma.

    Returns
    -------
    np.ndarray
        2D density map of shape (height, width) with dtype float32.
    """
    density = np.zeros((height, width), dtype=np.float32)
    if points_xy is None or len(points_xy) == 0:
        return density

    pts_arr = np.asarray(points_xy, dtype=np.float64)
    if pts_arr.ndim != 2 or pts_arr.shape[1] < 2:
        return density

    # Filter out non-finite coordinates and points outside [0, width) x [0, height)
    valid_points = []
    for raw_pt in pts_arr:
        x_val, y_val = float(raw_pt[0]), float(raw_pt[1])
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            continue
        px, py = int(x_val), int(y_val)
        if 0 <= px < width and 0 <= py < height:
            valid_points.append((px, py))

    num_pts = len(valid_points)
    if num_pts == 0:
        return density

    coords = np.asarray(valid_points, dtype=np.float64)

    # Estimate per-point bandwidth sigma_i = beta * mean(d_{i,1..m})
    if num_pts == 1:
        sigmas = np.full(1, max(min_sigma, 0.125 * (height + width)), dtype=np.float64)
    else:
        query_k = min(k_neighbors + 1, num_pts)
        spatial_index = cKDTree(coords)
        nn_dists, _ = spatial_index.query(coords, k=query_k)
        if nn_dists.ndim == 1:
            neighbor_mean = nn_dists
        else:
            neighbor_mean = np.mean(nn_dists[:, 1:], axis=1)
        sigmas = np.maximum(beta * neighbor_mean, min_sigma)

    # Accumulate normalized 2D Gaussian kernels within bounded local windows
    for idx in range(num_pts):
        cx, cy = int(coords[idx, 0]), int(coords[idx, 1])
        sigma = max(float(sigmas[idx]), min_sigma)
        radius = max(1, int(np.ceil(truncate_stds * sigma)))

        x0 = max(0, cx - radius)
        x1 = min(width, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(height, cy + radius + 1)

        dy = np.arange(y0 - cy, y1 - cy, dtype=np.float32)
        dx = np.arange(x0 - cx, x1 - cx, dtype=np.float32)
        inv_two_var = np.float32(1.0 / (2.0 * sigma * sigma))

        ky = np.exp(-(dy * dy) * inv_two_var)
        kx = np.exp(-(dx * dx) * inv_two_var)
        kernel = np.outer(ky, kx)

        kernel_sum = float(kernel.sum())
        if kernel_sum > 0.0:
            kernel /= np.float32(kernel_sum)

        density[y0:y1, x0:x1] += kernel

    return density


def gaussian_filter_density(gt: np.ndarray) -> np.ndarray:
    """
    Compatibility wrapper accepting a 2D point-count / binary array `gt` of shape (H, W)
    and delegating to the clean-room `build_geometry_adaptive_density` implementation.
    """
    gt_arr = np.asarray(gt)
    if gt_arr.ndim != 2:
        raise ValueError(f"Expected 2D array for gt, got shape {gt_arr.shape}")
    height, width = gt_arr.shape
    ys, xs = np.nonzero(gt_arr > 0)
    if len(xs) == 0:
        return np.zeros((height, width), dtype=np.float32)
    points_xy = np.column_stack((xs, ys))
    return build_geometry_adaptive_density(height, width, points_xy)


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

    density = build_geometry_adaptive_density(h, w, pts)

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
