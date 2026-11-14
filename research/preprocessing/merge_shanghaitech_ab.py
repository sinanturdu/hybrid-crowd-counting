import os
import shutil

# ✅ Kaggle’dan indirilen ShanghaiTech KÖK klasörü (SENİN YOLUN)
DATASET_IN = os.getenv("SHANGHAITECH_RAW_DIR", "datasets/ShanghaiTech_raw")

# ✅ CSRNet için hedef klasör
DATASET_OUT = os.getenv("SHANGHAITECH_OUT_DIR", "datasets/ShanghaiTech")

def copy_with_prefix(src_img_dir, src_gt_dir, dst_img_dir, dst_gt_dir, prefix):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_gt_dir, exist_ok=True)

    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(".jpg")]

    copied = 0
    for img_name in img_files:
        base = os.path.splitext(img_name)[0]
        src_img = os.path.join(src_img_dir, img_name)

        gt_name = f"GT_{base}.mat"
        src_gt = os.path.join(src_gt_dir, gt_name)

        if not os.path.exists(src_gt):
            continue

        new_img_name = f"{prefix}_{img_name}"
        new_gt_name = f"GT_{prefix}_{base}.mat"

        shutil.copy2(src_img, os.path.join(dst_img_dir, new_img_name))
        shutil.copy2(src_gt, os.path.join(dst_gt_dir, new_gt_name))
        copied += 1

    return copied

def process(part):
    base = os.path.join(DATASET_IN, part)

    train_img = os.path.join(base, "train_data", "images")
    train_gt  = os.path.join(base, "train_data", "ground_truth")
    test_img  = os.path.join(base, "test_data", "images")
    test_gt   = os.path.join(base, "test_data", "ground_truth")

    out_train_img = os.path.join(DATASET_OUT, "train", "images")
    out_train_gt  = os.path.join(DATASET_OUT, "train", "ground_truth")
    out_test_img  = os.path.join(DATASET_OUT, "test", "images")
    out_test_gt   = os.path.join(DATASET_OUT, "test", "ground_truth")

    print(f"\n▶ {part} işleniyor...")
    n1 = copy_with_prefix(train_img, train_gt, out_train_img, out_train_gt, part)
    n2 = copy_with_prefix(test_img, test_gt, out_test_img, out_test_gt, part)
    print(f"   Train: {n1} | Test: {n2}")

def main():
    process("part_A")
    process("part_B")
    print("\n🎉 ShanghaiTech A + B başarıyla birleştirildi")

if __name__ == "__main__":
    main()
