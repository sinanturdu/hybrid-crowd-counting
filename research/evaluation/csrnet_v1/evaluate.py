import torch
from torch.utils.data import DataLoader
from ucf_qnrf.dataset import ShanghaiTechDataset
from ucf_qnrf.model import CSRNet
import numpy as np

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Kullanılan cihaz:", device)

# ======================
# DATASET
# ======================
dataset_root = "../datasets/ShanghaiTech"
test_dataset = ShanghaiTechDataset(dataset_root, split="test")

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# ======================
# MODEL
# ======================
model = CSRNet()
model.load_state_dict(torch.load("csrnet_shanghaitech.pth", map_location=device))
model.to(device)
model.eval()

# ======================
# METRICS
# ======================
mae = 0.0
mse = 0.0
num_samples = len(test_loader)

# ======================
# EVALUATION LOOP
# ======================
with torch.no_grad():
    for imgs, gts in test_loader:
        imgs = imgs.to(device)
        gts = gts.to(device)

        preds = model(imgs)

        gt_count = gts.sum().item()
        pred_count = preds.sum().item()

        diff = abs(gt_count - pred_count)

        mae += diff
        mse += diff ** 2

# ======================
# RESULTS
# ======================
mae /= num_samples
mse /= num_samples
rmse = np.sqrt(mse)

print("\n📊 CSRNet TEST SONUÇLARI")
print(f"📌 MAE  : {mae:.2f}")
print(f"📌 MSE  : {mse:.2f}")
print(f"📌 RMSE : {rmse:.2f}")
