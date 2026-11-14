import os
import cv2
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter


class ShanghaiTechDataset(Dataset):
    """
    CSRNet için SAĞLAM ShanghaiTech Dataset
    - recursion YOK
    - controlled sampling
    - patch-based
    """

    def __init__(self, root, crop_size=512, downscale=8, sigma=4, max_try=20):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.gt_dir = os.path.join(root, "ground_truth")

        self.crop_size = crop_size
        self.downscale = downscale
        self.sigma = sigma
        self.max_try = max_try

        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(".jpg")
        ])

        if len(self.images) == 0:
            raise RuntimeError("train/images boş")

        print(f"📊 Toplam eğitim görüntüsü: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def _read_image(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def _points_to_density(self, points):
        cs = self.crop_size
        dh, dw = cs // self.downscale, cs // self.downscale
        density = np.zeros((dh, dw), dtype=np.float32)

        for x, y in points:
            xx = int(x / self.downscale)
            yy = int(y / self.downscale)
            if 0 <= xx < dw and 0 <= yy < dh:
                density[yy, xx] += 1.0

        return gaussian_filter(density, sigma=self.sigma)

    def _crop_centered(self, img, cx, cy):
        h, w, _ = img.shape
        cs = self.crop_size
        x0 = int(np.clip(cx - cs // 2, 0, w - cs))
        y0 = int(np.clip(cy - cs // 2, 0, h - cs))
        return img[y0:y0 + cs, x0:x0 + cs], x0, y0

    def __getitem__(self, idx):
        for _ in range(self.max_try):

            img_name = self.images[idx]
            img_path = os.path.join(self.img_dir, img_name)
            gt_path = os.path.join(
                self.gt_dir,
                img_name.replace("IMG_", "GT_IMG_").replace(".jpg", ".mat")
            )

            img = self._read_image(img_path)
            if img is None:
                idx = np.random.randint(len(self))
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            h, w, _ = img.shape

            mat = sio.loadmat(gt_path)

            # -------- Part A --------
            if "image_info" in mat:
                points = mat["image_info"][0, 0][0, 0][0]
                if len(points) == 0:
                    idx = np.random.randint(len(self))
                    continue

                p = points[np.random.randint(len(points))]
                cx, cy = int(p[0]), int(p[1])

                img_crop, x0, y0 = self._crop_centered(img, cx, cy)

                pts = points.copy()
                pts[:, 0] -= x0
                pts[:, 1] -= y0

                mask = (
                    (pts[:, 0] >= 0) & (pts[:, 0] < self.crop_size) &
                    (pts[:, 1] >= 0) & (pts[:, 1] < self.crop_size)
                )
                pts = pts[mask]

                if len(pts) == 0:
                    idx = np.random.randint(len(self))
                    continue

                density = self._points_to_density(pts)

            # -------- Part B --------
            elif "density" in mat:
                density_full = mat["density"]
                if density_full.sum() < 1:
                    idx = np.random.randint(len(self))
                    continue

                ys, xs = np.where(density_full > 0)
                sel = np.random.randint(len(xs))
                cx, cy = xs[sel], ys[sel]

                img_crop, x0, y0 = self._crop_centered(img, cx, cy)
                density_crop = density_full[y0:y0 + self.crop_size,
                                            x0:x0 + self.crop_size]

                density = cv2.resize(
                    density_crop,
                    (self.crop_size // self.downscale,
                     self.crop_size // self.downscale),
                    interpolation=cv2.INTER_AREA
                )
                density *= density_crop.sum() / max(density.sum(), 1e-6)

            else:
                idx = np.random.randint(len(self))
                continue

            # ✅ BAŞARILI SAMPLE
            img_t = torch.from_numpy(img_crop).permute(2, 0, 1).contiguous()
            den_t = torch.from_numpy(density).unsqueeze(0).contiguous()
            return img_t, den_t

        # 🔴 fallback: center crop (çok nadir)
        img_crop = img[:self.crop_size, :self.crop_size]
        density = np.zeros((self.crop_size // self.downscale,
                            self.crop_size // self.downscale), dtype=np.float32)
        return (
            torch.from_numpy(img_crop).permute(2, 0, 1).contiguous(),
            torch.from_numpy(density).unsqueeze(0)
        )
