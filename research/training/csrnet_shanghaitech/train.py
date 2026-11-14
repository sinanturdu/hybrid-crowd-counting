import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from models.csrnet import CSRNet
from datasets.shanghaitech import ShanghaiTech


# ======================================================
# SUM-PRESERVING RESIZE
# ======================================================
def resize_density_preserve_sum(dens, out_hw):
    old_sum = dens.sum(dim=(2, 3), keepdim=True)
    dens_ds = F.interpolate(dens, size=out_hw, mode="area")
    new_sum = dens_ds.sum(dim=(2, 3), keepdim=True)
    dens_ds = dens_ds * (old_sum / (new_sum + 1e-8))
    return dens_ds


# ======================================================
# GT SHARPEN
# ======================================================
def sharpen_density(dens, alpha=0.30):
    blurred = F.avg_pool2d(dens, kernel_size=3, stride=1, padding=1)
    sharp = dens + alpha * (dens - blurred)
    return torch.clamp(sharp, min=0.0)


# ======================================================
# BACKGROUND SUPPRESSION LOSS
# ======================================================
def background_suppression_loss(preds, gt_ds, bg_thresh=1e-6, alpha=2.0):
    bg_mask = (gt_ds <= bg_thresh).float()
    bg_pred = preds * bg_mask
    return (bg_pred ** alpha).mean()


# ======================================================
# 🔥 A / B SCENE CLASSIFIER (TRAIN TIME)
# ======================================================
def classify_scene_from_gt(gt_ds):
    """
    gt_ds: (B,1,H,W)
    return: scene_weight tensor (B,)
    """
    gt_count = gt_ds.sum(dim=(1, 2, 3))
    active_pixels = (gt_ds > 1e-6).sum(dim=(1, 2, 3))
    area_ratio = active_pixels.float() / (gt_ds.shape[2] * gt_ds.shape[3])

    # Dense (A): yüksek count + küçük alan
    is_dense = (gt_count > 400) & (area_ratio < 0.35)

    # A -> 1.0 | B -> 0.5
    scene_weight = torch.where(
        is_dense,
        torch.ones_like(gt_count),
        torch.full_like(gt_count, 0.5)
    )

    return scene_weight.detach()


# ======================================================
# EVALUATION
# ======================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae, mse, n = 0.0, 0.0, 0

    for imgs, dens, _ in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device)
        dens = dens.to(device)

        preds = torch.clamp(F.relu(model(imgs)), max=5.0)

        gt_ds = resize_density_preserve_sum(dens, preds.shape[-2:])
        gt_ds = sharpen_density(gt_ds)

        gt_count = gt_ds.sum(dim=(1,2,3))
        pr_count = preds.sum(dim=(1,2,3))

        diff = torch.abs(pr_count - gt_count)
        mae += diff.sum().item()
        mse += (diff ** 2).sum().item()
        n += imgs.size(0)

    return mae / n, math.sqrt(mse / n)


# ======================================================
# MAIN
# ======================================================
def main():
    DATA_ROOT = os.getenv("SHANGHAITECH_ROOT", "datasets/ShanghaiTech")

    EPOCHS = 120
    BATCH = 8
    LR = 5e-6
    CROP = 384
    NUM_WORKERS = 0

    SAVE_DIR = "weights"
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Device:", device)

    # ==================================================
    # DATASETS (A + B)
    # ==================================================
    train_A = ShanghaiTech(DATA_ROOT, part="A", split="train",
                           train_crop_size=CROP, train_random_crop=True)

    train_B = ShanghaiTech(DATA_ROOT, part="B", split="train",
                           train_crop_size=CROP, train_random_crop=True)

    test_A = ShanghaiTech(DATA_ROOT, part="A", split="test",
                          train_crop_size=CROP, train_random_crop=False)

    test_B = ShanghaiTech(DATA_ROOT, part="B", split="test",
                          train_crop_size=CROP, train_random_crop=False)

    train_ds = ConcatDataset([train_A, train_B])
    test_ds  = ConcatDataset([test_A, test_B])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    model = CSRNet(load_pretrained_vgg=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    pixel_loss_fn = nn.MSELoss()

    best_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Backbone warmup
        for p in model.frontend.parameters():
            p.requires_grad = epoch > 5

        BG_W = 0.02 if epoch <= 10 else 0.05 if epoch <= 30 else 0.08
        COUNT_W_BASE = 0.005

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, dens, _ in pbar:
            imgs = imgs.to(device)
            dens = dens.to(device)

            preds = torch.clamp(F.relu(model(imgs)), max=5.0)

            gt_ds = resize_density_preserve_sum(dens, preds.shape[-2:])
            gt_ds = sharpen_density(gt_ds)

            # ===============================
            # 🔥 A / B AWARE WEIGHTS
            # ===============================
            scene_w = classify_scene_from_gt(gt_ds)  # (B,)

            pixel_loss = pixel_loss_fn(preds, gt_ds)

            gt_count = gt_ds.sum(dim=(1,2,3))
            pr_count = preds.sum(dim=(1,2,3))
            abs_diff = torch.abs(pr_count - gt_count)

            count_loss = torch.mean(
                scene_w * (abs_diff / (gt_count + 1.0))
            )

            bg_loss = background_suppression_loss(preds, gt_ds)

            loss = (
                pixel_loss +
                COUNT_W_BASE * count_loss +
                BG_W * torch.mean(scene_w * bg_loss)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                loss=float(loss.item()),
                A_ratio=float((scene_w > 0.9).float().mean())
            )

        mae, rmse = evaluate(model, test_loader, device)
        print(f"✅ Epoch {epoch} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "mae": mae},
            os.path.join(SAVE_DIR, "csrnet_AB_last.pth")
        )

        if mae < best_mae:
            best_mae = mae
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "mae": mae},
                os.path.join(SAVE_DIR, "csrnet_AB_best.pth")
            )
            print("🏆 Best model saved")

    print("🎉 Training finished | Best MAE:", best_mae)


if __name__ == "__main__":
    main()
