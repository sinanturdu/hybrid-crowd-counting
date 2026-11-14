import os
import cv2
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree


class UCFQNRFDataset(Dataset):
    def __init__(
        self,
        root,
        split="Train",
        downscale=8,
        max_size=1024      # 🔥 OOM ENGELLEYİCİ
    ):
        self.root = root
        self.split = split
        self.downscale = downscale
        self.max_size = max_size

        self.img_dir = os.path.join(root, split)
        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".jpg")
        ])

        if len(self.images) == 0:
            raise RuntimeError(f"Dataset boş: {self.img_dir}")

    def __len__(self):
        return len(self.images)

    # =========================
    # LOAD POINTS
    # =========================
    def _load_points(self, mat_path):
        mat = sio.loadmat(mat_path)
        if "annPoints" not in mat:
            raise KeyError(f"annPoints yok: {mat_path}")

        pts = mat["annPoints"].astype(np.float32)

        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError(f"Geçersiz annPoints formatı: {pts.shape}")

        return pts[:, :2]

    # =========================
    # RESIZE (OOM SAFE)
    # =========================
    def _resize_if_needed(self, img, points):
        h, w, _ = img.shape
        max_dim = max(h, w)

        if max_dim <= self.max_size:
            return img, points

        scale = self.max_size / max_dim
        new_h = int(h * scale)
        new_w = int(w * scale)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pts = points.copy()
        pts[:, 0] *= scale
        pts[:, 1] *= scale

        return img, pts

    # =====================================================
    # 🔥 GEOMETRY-ADAPTIVE DENSITY (CSRNet OFFICIAL STYLE)
    # =====================================================
    def _make_density(self, points, h, w):
        dh = max(1, h // self.downscale)
        dw = max(1, w // self.downscale)

        density = np.zeros((dh, dw), dtype=np.float32)

        if len(points) == 0:
            return density

        tree = KDTree(points)
        distances, _ = tree.query(points, k=min(4, len(points)))

        for i, (x, y) in enumerate(points):
            ix = int(x / self.downscale)
            iy = int(y / self.downscale)

            if ix < 0 or ix >= dw or iy < 0 or iy >= dh:
                continue

            if len(points) > 3:
                sigma = distances[i][1:4].mean()
            else:
                sigma = distances[i][1] if len(points) > 1 else 15.0

            sigma = max(sigma / self.downscale, 1.0)

            tmp = np.zeros((dh, dw), dtype=np.float32)
            tmp[iy, ix] = 1.0

            density += gaussian_filter(tmp, sigma=sigma)

        return density

    # =========================
    # GET ITEM
    # =========================
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mat_path = img_path.replace(".jpg", "_ann.mat")

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Okunamadı: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        points = self._load_points(mat_path)

        # 🔥 resize (OOM koruma)
        img, points = self._resize_if_needed(img, points)

        h, w, _ = img.shape
        density = self._make_density(points, h, w)

        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        den_t = torch.from_numpy(density).unsqueeze(0).contiguous()

        return img_t, den_t
