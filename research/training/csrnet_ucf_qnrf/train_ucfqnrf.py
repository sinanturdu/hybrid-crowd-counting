import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math

from model import CSRNet
from dataset_ucfqnrf import UCFQNRFDataset

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Kullanılan cihaz:", device)

torch.backends.cudnn.benchmark = True

# =========================
# DATASET
# =========================
dataset_root = "./"

train_dataset = UCFQNRFDataset(
    root=dataset_root,
    split="Train"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,                 # CSRNet için zorunlu
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# =========================
# MODEL
# =========================
model = CSRNet().to(device)

criterion = nn.MSELoss()

# 🔥 ÇOK DÜŞÜK LR (CSRNet şart)
optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)

epochs = 30

# =========================
# TRAIN
# =========================
for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    abs_errors = []
    sq_errors = []

    print(f"\n🚀 Epoch {epoch+1}/{epochs}")

    for i, (imgs, gts) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        preds = model(imgs)

        loss = criterion(preds, gts)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 🔥 STABİLİTE
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()

        # =========================
        # METRICS (LOG ONLY)
        # =========================
        gt_count = gts.sum().item()
        pred_count = torch.relu(preds).sum().item()

        diff = abs(pred_count - gt_count)

        abs_errors.append(diff)
        sq_errors.append(diff ** 2)

        if i % 10 == 0:
            print(
                f"  🔹 Batch {i:4d} | "
                f"Loss: {loss.item():.6f} | "
                f"GT: {gt_count:.1f} | "
                f"Pred: {pred_count:.1f} | "
                f"Diff: {diff:.1f}"
            )

    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors))

    print(f"✅ Epoch Ortalama Loss : {total_loss / len(train_loader):.6f}")
    print(f"📊 MAE  : {mae:.2f}")
    print(f"📊 RMSE : {rmse:.2f}")

# =========================
# SAVE
# =========================
torch.save(model.state_dict(), "csrnet_ucf_qnrf_best.pth")
print("💾 Model kaydedildi: csrnet_ucf_qnrf_best.pth")
