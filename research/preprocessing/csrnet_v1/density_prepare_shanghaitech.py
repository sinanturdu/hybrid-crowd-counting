import shutil
from pathlib import Path

# ===== PATHS =====
BASE = Path("datasets/ShanghaiTech")
OUT  = Path("datasets/shanghaitech_combined")

def copy_split(part_name):
    part = BASE / part_name

    for split in ["train_data", "test_data"]:
        img_src = part / split / "images"
        gt_src  = part / split / "ground-truth"

        out_split = "train" if split == "train_data" else "test"
        img_dst = OUT / out_split / "images"
        gt_dst  = OUT / out_split / "ground_truth"

        img_dst.mkdir(parents=True, exist_ok=True)
        gt_dst.mkdir(parents=True, exist_ok=True)

        for img in img_src.glob("*.jpg"):
            new_name = f"{part_name[-1]}_{img.name}"
            shutil.copy(img, img_dst / new_name)

            gt_name = f"GT_{img.stem}.mat"
            gt_file = gt_src / gt_name
            if gt_file.exists():
                shutil.copy(gt_file, gt_dst / f"{part_name[-1]}_{gt_name}")
            else:
                print(f"❌ GT bulunamadı: {gt_name}")

    print(f"✅ {part_name} tamamlandı")

# ===== RUN =====
copy_split("part_A")
copy_split("part_B")

print("\n🎉 ShanghaiTech A+B başarıyla birleştirildi")
