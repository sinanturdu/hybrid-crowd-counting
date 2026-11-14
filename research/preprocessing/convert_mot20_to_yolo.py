import os
import cv2

# ================================
# MOT20 dataset paths
# ================================
MOT_ROOT = os.getenv("MOT20_ROOT", "datasets/MOT20")
OUTPUT_ROOT = os.getenv("MOT20_YOLO_OUT", "datasets/MOT20_YOLO")

# Çıktı klasörlerini oluştur
os.makedirs(f"{OUTPUT_ROOT}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/labels/train", exist_ok=True)

TRAIN_SEQUENCES = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]

# ================================
# DÖNÜŞTÜRME İŞLEMİ BAŞLIYOR
# ================================
for seq in TRAIN_SEQUENCES:
    seq_path = os.path.join(MOT_ROOT, "train", seq)
    img_dir = os.path.join(seq_path, "img1")
    gt_file = os.path.join(seq_path, "gt", "gt.txt")

    print(f"🔵 Processing {seq}...")

    with open(gt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        frame, track_id, x, y, w, h, conf, cls, vis = line.strip().split(",")
        frame = int(frame)

        img_path = os.path.join(img_dir, f"{frame:06d}.jpg")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]

        # YOLO formatı (normalize)
        xc = (float(x) + float(w) / 2) / W
        yc = (float(y) + float(h) / 2) / H
        ww = float(w) / W
        hh = float(h) / H

        # Çıktı dosya isimleri
        img_out_name = f"{seq}_{frame:06d}.jpg"
        label_out_name = f"{seq}_{frame:06d}.txt"

        # Etiketi kaydet
        with open(f"{OUTPUT_ROOT}/labels/train/{label_out_name}", "a") as lf:
            lf.write(f"0 {xc} {yc} {ww} {hh}\n")

        # Görüntüyü kaydet (kopyala)
        out_img_path = f"{OUTPUT_ROOT}/images/train/{img_out_name}"
        if not os.path.exists(out_img_path):
            cv2.imwrite(out_img_path, img)

print("\n✅ MOT20 → YOLO dönüştürme TAMAMLANDI!")
