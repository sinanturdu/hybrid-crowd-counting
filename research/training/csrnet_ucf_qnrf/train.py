import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UCFQNRFDataset
from model import CSRNet

ROOT = os.getenv("UCF_QNRF_ROOT", "datasets/ucf_qnrf")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 Device:", device)

train_ds = UCFQNRFDataset(ROOT, "Train")
test_ds  = UCFQNRFDataset(ROOT, "Test")

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

model = CSRNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

os.makedirs("models", exist_ok=True)

EPOCHS = 10
best_mae = 1e9

for epoch in range(EPOCHS):
    model.train()
    loss_sum = 0

    for img, den in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        img, den = img.to(device), den.to(device)
        pred = model(img)

        loss = criterion(pred, den)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    model.eval()
    mae = 0
    with torch.no_grad():
        for img, den in test_loader:
            img, den = img.to(device), den.to(device)
            pred = model(img)
            mae += abs(pred.sum().item() - den.sum().item())

    mae /= len(test_loader)

    print(f"✅ Epoch {epoch+1} | Loss={loss_sum/len(train_loader):.6f} | MAE={mae:.2f}")

    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), "models/csrnet_best.pth")
        print("🏆 Best model kaydedildi")
