import os
import shutil

SRC_ROOT = os.getenv("SHANGHAITECH_SRC_DIR", "datasets/ShanghaiTech")
DST_ROOT = r"datasets\ShanghaiTech"

def process_part(part):
    for split in ["train_data", "test_data"]:
        src_img = os.path.join(SRC_ROOT, part, split, "images")
        src_gt  = os.path.join(SRC_ROOT, part, split, "ground-truth")

        dst_split = "train" if split == "train_data" else "test"
        dst_img = os.path.join(DST_ROOT, dst_split, "images")
        dst_gt  = os.path.join(DST_ROOT, dst_split, "ground_truth")

        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_gt, exist_ok=True)

        for fname in os.listdir(src_img):
            if not fname.endswith(".jpg"):
                continue

            new_img_name = f"{part}_{fname}"
            shutil.copy(
                os.path.join(src_img, fname),
                os.path.join(dst_img, new_img_name)
            )

            gt_name = "GT_" + fname.replace(".jpg", ".mat")
            src_gt_path = os.path.join(src_gt, gt_name)

            if os.path.exists(src_gt_path):
                new_gt_name = f"{part}_{gt_name}"
                shutil.copy(
                    src_gt_path,
                    os.path.join(dst_gt, new_gt_name)
                )
            else:
                print(f"⚠️ GT yok: {new_img_name}")

if __name__ == "__main__":
    process_part("part_A")
    process_part("part_B")
    print("\n🎉 ShanghaiTech A + B CSRNet formatında TEMİZ kuruldu!")
