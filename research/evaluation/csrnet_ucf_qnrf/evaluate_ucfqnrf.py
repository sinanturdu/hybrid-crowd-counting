import torch
import numpy as np
from torch.utils.data import DataLoader
from model import CSRNet
from dataset_ucfqnrf import UCFQNFRDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = UCFQNFRDataset(
    root="./",
    split="Test",
    train=False
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = CSRNet().to(device)
model.load_state_dict(torch.load("csrnet_ucf_qnrf_best.pth", map_location=device))
model.eval()

mae, mse = 0, 0

print("📊 TEST BAŞLADI")

with torch.no_grad():
    for i, (img, gt) in enumerate(test_loader):
        img = img.to(device)
        gt_count = gt.sum().item()

        pred = model(img)
        pred_count = pred.sum().item()

        diff = abs(gt_count - pred_count)
        mae += diff
        mse += diff ** 2

        if i % 10 == 0:
            print(f"[{i}/{len(test_loader)}] GT: {gt_count:.1f} | Pred: {pred_count:.1f} | Diff: {diff:.1f}")

mae /= len(test_loader)
rmse = np.sqrt(mse / len(test_loader))

print("\n==============================")
print("📊 UCF-QNRF TEST SONUÇLARI")
print(f"📌 MAE  : {mae:.2f}")
print(f"📌 RMSE : {rmse:.2f}")
print("==============================")
