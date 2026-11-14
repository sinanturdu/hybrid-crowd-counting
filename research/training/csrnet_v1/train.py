import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.csrnet import CSRNet
from utils.dataset import ShanghaiTechDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🔥 Kullanılan cihaz:", device)

    train_dataset = ShanghaiTechDataset(
        "datasets/ShanghaiTech/part_A/train_data/images",
        "datasets/ShanghaiTech/part_A/train_data/ground_truth"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    print(f"📊 Toplam eğitim görüntüsü: {len(train_dataset)}")

    model = CSRNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        print(f"\n🚀 Epoch {epoch+1}/{epochs}")

        for i, (imgs, gts) in enumerate(train_loader):
            imgs = imgs.to(device)
            gts = gts.to(device)

            preds = model(imgs)

            # 🔧 GT'yi pred boyutuna getir (CSRNet standardı)
            gts = F.interpolate(
                gts,
                size=preds.shape[2:],
                mode="bilinear",
                align_corners=False
            )

            loss = criterion(preds, gts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(
                    f"  🔹 Batch {i} | Loss: {loss.item():.6f} | "
                    f"GT SUM: {gts.sum().item():.2f} | "
                    f"PRED SUM: {preds.sum().item():.2f}"
                )

        print(f"✅ Epoch {epoch+1} Ortalama Loss: {epoch_loss / len(train_loader):.6f}")

    torch.save(model.state_dict(), "csrnet_partA.pth")
    print("\n💾 Model kaydedildi: csrnet_partA.pth")


if __name__ == "__main__":
    main()
