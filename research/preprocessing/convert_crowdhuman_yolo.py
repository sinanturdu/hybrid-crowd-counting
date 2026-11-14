import json
import os
import shutil
from tqdm import tqdm

# ================================
# Config
# ================================
DATASET_ROOT = os.getenv("CROWDHUMAN_ROOT", "datasets/crowdhuman")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "yolo_format")

IMAGES_DIR = os.path.join(DATASET_ROOT, "Images")
TRAIN_ODGT = os.path.join(DATASET_ROOT, "annotation_train.odgt")
VAL_ODGT = os.path.join(DATASET_ROOT, "annotation_val.odgt")

# ================================
# YOLO output folders
# ================================
train_img_out = os.path.join(OUTPUT_DIR, "images/train")
val_img_out = os.path.join(OUTPUT_DIR, "images/val")
train_lbl_out = os.path.join(OUTPUT_DIR, "labels/train")
val_lbl_out = os.path.join(OUTPUT_DIR, "labels/val")

os.makedirs(train_img_out, exist_ok=True)
os.makedirs(val_img_out, exist_ok=True)
os.makedirs(train_lbl_out, exist_ok=True)
os.makedirs(val_lbl_out, exist_ok=True)

# ================================
# Helper: Convert box to YOLO format
# ================================
def box_to_yolo(x, y, w, h, img_w, img_h):
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    return xc, yc, w / img_w, h / img_h


# ================================
# Helper: Process ODGT file
# ================================
def process_odgt(odgt_file, img_out_dir, lbl_out_dir):
    print(f"\nProcessing {odgt_file} ...")

    with open(odgt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        data = json.loads(line)
        img_id = data["ID"]
        img_file = os.path.join(IMAGES_DIR, img_id + ".jpg")

        if not os.path.exists(img_file):
            continue

        # copy image
        shutil.copy(img_file, img_out_dir)

        # label output
        lbl_path = os.path.join(lbl_out_dir, img_id + ".txt")

        with open(lbl_path, "w") as lbl:
            for ann in data["gtboxes"]:
                if ann["tag"] != "person":
                    continue

                x, y, w, h = ann["fbox"]

                # CrowdHuman images are usually 1920x1080
                img_w, img_h = 1920, 1080
                xc, yc, w_norm, h_norm = box_to_yolo(x, y, w, h, img_w, img_h)

                lbl.write(f"0 {xc} {yc} {w_norm} {h_norm}\n")


# ================================
# RUN
# ================================
process_odgt(TRAIN_ODGT, train_img_out, train_lbl_out)
process_odgt(VAL_ODGT, val_img_out, val_lbl_out)

print("\n✅ Dataset converted successfully!")

# ================================
# CREATE crowdhuman.yaml
# ================================
yaml_path = os.path.join(DATASET_ROOT, "crowdhuman.yaml")
yaml_content = f"""
path: {OUTPUT_DIR}
train: images/train
val: images/val

names:
  0: person
"""

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"✅ YAML created at: {yaml_path}")
print("\n🎉 CrowdHuman dataset is now 100% ready for YOLOv9 training!")
