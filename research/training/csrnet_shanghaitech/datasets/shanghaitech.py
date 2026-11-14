import os
import glob
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# ======================================================
# NORMALIZATION
# ======================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ======================================================
# UTILS
# ======================================================
def _read_density_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        density = np.array(f["density"], dtype=np.float32)
    return density

def _preserve_sum_resize(density, new_w, new_h):
    """
    Resize density map and preserve total count (sum).
    CPU-safe resize via torch interpolate.
    """
    old_sum = float(density.sum())
    if new_w <= 0 or new_h <= 0:
        raise ValueError("Invalid resize target")

    # Always do resize on CPU to avoid any device issues inside DataLoader workers
    d = torch.from_numpy(density).unsqueeze(0).unsqueeze(0)  # (1,1,H,W) CPU
    d = F.interpolate(d, size=(new_h, new_w), mode="bilinear", align_corners=False)
    d = d.squeeze(0).squeeze(0).contiguous().cpu().numpy().astype(np.float32)

    new_sum = float(d.sum())
    if new_sum > 0:
        d *= (old_sum / new_sum)
    return d.astype(np.float32)

# ======================================================
# DATASET
# ======================================================
class ShanghaiTech(Dataset):
    """
    root: .../ShanghaiTech
    part: "A" or "B"
    split: "train" or "test"

    ✅ ÖNEMLİ:
    - GT density map ORİJİNAL çözünürlükte döner (H,W)
    - Train'de SABİT crop (örn 384) => batch safe
    - train.py'de model output upsample edilip loss alınır
    """

    def __init__(
        self,
        root,
        part="A",
        split="train",
        train_crop_size=384,
        train_random_crop=True,
        hflip=True
    ):
        self.root = root
        self.part = part.upper()
        self.split = split.lower()

        assert self.part in ["A", "B"]
        assert self.split in ["train", "test"]

        base = os.path.join(root, f"part_{self.part}", f"{self.split}_data")
        self.img_dir = os.path.join(base, "images")
        self.h5_dir = os.path.join(base, "ground-truth-h5")

        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No JPG found in {self.img_dir}")

        self.train_crop_size = int(train_crop_size)
        self.train_random_crop = bool(train_random_crop)
        self.hflip = bool(hflip)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.img_paths)

    def _img_to_h5(self, img_path):
        name = os.path.splitext(os.path.basename(img_path))[0]  # IMG_123
        return os.path.join(self.h5_dir, f"{name}.h5")

    def __getitem__(self, idx):
        # loop-safe fallback (never crash training)
        idx = int(idx) % len(self.img_paths)

        img_path = self.img_paths[idx]
        h5_path = self._img_to_h5(img_path)

        # h5 yoksa başka sample dene
        if not os.path.exists(h5_path):
            return self.__getitem__(idx + 1)

        # Unicode-safe image read
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__(idx + 1)

        img = np.array(img_pil).astype(np.float32) / 255.0
        H, W = img.shape[:2]

        # density
        try:
            density = _read_density_h5(h5_path)
        except Exception:
            return self.__getitem__(idx + 1)

        # density mismatch varsa düzelt
        if density.shape[0] != H or density.shape[1] != W:
            density = _preserve_sum_resize(density, W, H)

        # ======================================================
        # TRAIN AUGMENT (SABİT CROP)
        # ======================================================
        if self.split == "train":
            # flip
            if self.hflip and random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                density = np.ascontiguousarray(density[:, ::-1])

            cs = self.train_crop_size  # sabit crop

            # crop'tan küçükse sample atla
            if H < cs or W < cs:
                return self.__getitem__(idx + 1)

            if self.train_random_crop:
                y = random.randint(0, H - cs)
                x = random.randint(0, W - cs)
            else:
                y = (H - cs) // 2
                x = (W - cs) // 2

            img = img[y:y + cs, x:x + cs, :]
            density = density[y:y + cs, x:x + cs]
            H, W = cs, cs

        # ======================================================
        # TO TORCH
        # ======================================================
        img_uint8 = (img * 255.0).astype(np.uint8)
        img_t = self.to_tensor(img_uint8)  # (3,H,W)

        # ✅ GT density ORİJİNAL (H,W)
        den_t = torch.from_numpy(density).unsqueeze(0)  # (1,H,W)

        return img_t, den_t, img_path
