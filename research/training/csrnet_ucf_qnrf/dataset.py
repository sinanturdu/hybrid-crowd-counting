import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class UCFQNRFDataset(Dataset):
    def __init__(self, root, split="Train"):
        self.img_dir = os.path.join(root, split)
        self.den_dir = os.path.join(root, "density", split)

        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".jpg")])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = img_name.replace(".jpg", "")

        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        img = self.transform(img)

        density = np.load(os.path.join(self.den_dir, base + ".npy"))
        density = torch.from_numpy(density).unsqueeze(0)

        return img, density
