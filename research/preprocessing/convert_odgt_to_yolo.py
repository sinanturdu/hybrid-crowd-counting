import json
import os
from tqdm import tqdm
from PIL import Image

# ================================
# ABSOLUTE PATH (KESİN ÇALIŞIR)
# ================================
DATASET_DIR = os.getenv("CROWDHUMAN_ROOT", "datasets/crowdhuman")

TRAIN_ANNOT = os.path.join(DATASET_DIR, "annotation_train.odgt")
VAL_ANNOT   = os.path.join(DATASET_DIR, "annotation_val.odgt")

IMG_FOLDER       = os.path.join(DATASET_DIR, "Images")
VAL_IMG_FOLDER   = os.path.join(DATASET_DIR, "images_test")

OUT_TRAIN = os.path.join(DATASET_DIR, "labels", "train")
OUT_VAL   = os.path.join(DATASET_DIR, "labels", "val")

os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_VAL,  exist_ok=True)


def convert_box(box, img_w, img_h):
    x, y, w, h = box
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return xc, yc, w, h


def process_odgt(odgt_path, img_folder, out_folder):
    print(f"\nProcessing {odgt_path} ...")
    with open(odgt_path, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        data = json.loads(line)
        img_name = data["ID"] + ".jpg"
        img_path = os.path.join(img_folder, img_name)

        if not os.path.exists(img_path):
            continue

        # Görüntü boyutunu al
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except:
            continue

        label_path = os.path.join(out_folder, img_name.replace(".jpg", ".txt"))

        with open(label_path, "w") as out:
            for ann in data["gtboxes"]:
                if ann["tag"] != "person":
                    continue

                box = ann["fbox"]
                xc, yc, w, h = convert_box(box, img_w, img_h)

                out.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# İşlemleri başlat
process_odgt(TRAIN_ANNOT, IMG_FOLDER, OUT_TRAIN)
process_odgt(VAL_ANNOT, VAL_IMG_FOLDER, OUT_VAL)

print("\nDönüştürme TAMAMLANDI!")
