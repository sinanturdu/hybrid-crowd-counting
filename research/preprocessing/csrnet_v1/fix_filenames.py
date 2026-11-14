import os

ROOT = "datasets/ShanghaiTech"

def fix_split(split):
    img_dir = os.path.join(ROOT, split, "images")
    gt_dir = os.path.join(ROOT, split, "ground_truth")

    for fname in os.listdir(img_dir):
        if fname.startswith("part_"):
            new_name = fname.split("_", 1)[1]
            os.rename(
                os.path.join(img_dir, fname),
                os.path.join(img_dir, new_name)
            )

    for fname in os.listdir(gt_dir):
        if fname.startswith("part_"):
            new_name = "GT_" + fname.split("GT_", 1)[1]
            os.rename(
                os.path.join(gt_dir, fname),
                os.path.join(gt_dir, new_name)
            )

for split in ["train", "test"]:
    fix_split(split)

print("✅ Dosya isimleri düzeltildi")
