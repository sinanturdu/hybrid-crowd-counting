# Models, Experimental Results & Excluded Large Assets Inventory

## 1. Runtime Models Used by the Application (`backend/main.py`)

All three runtime model checkpoints in `backend/models/` have been cryptographically verified (MD5/SHA-256) against their originating research training runs in `bitirme_projesi`:

| Registry Key | Filename | Size | MD5 Prefix | Architecture | Source Training Run |
|---|---|---|---|---|---|
| `best` (default) | `backend/models/best.pt` | `22,518,169 B` (~22.5 MB) | `A3A32B3DF5` | Ultralytics YOLOv8s (`nc=1`, `person`) | `runs/detect/train14/weights/best.pt` |
| `best2` | `backend/models/best2.pt` | `22,516,900 B` (~22.5 MB) | `CB238468C7` | Ultralytics YOLOv8s (`nc=1`, `person`) | `yolo8/runs/detect/train/weights/best2.pt` |
| `csrnet_ab` | `backend/models/csrnet_AB_best.pth` | `65,066,362 B` (~65.1 MB) | `F34C96BDCE` | PyTorch CSRNet (VGG16 front-end + dilated conv back-end) | `ShanghaiTech/weights/csrnet_AB_best.pth` |

### Why Model Weights Are Excluded from Git
- Together, the three runtime weights total **~105.0 MB** (and `csrnet_AB_best.pth` alone is `65.1 MB`, above GitHub's 50 MB warning threshold).
- The weights were trained on datasets with varying redistribution terms (`reid` MIT, Roboflow CrowdHuman CC BY 4.0, ShanghaiTech academic/research terms).
- **Recommended Distribution Strategy:** Attach `best.pt`, `best2.pt`, and `csrnet_AB_best.pth` to a **GitHub Release** (or host on Hugging Face Hub / institutional storage) once redistribution permissions are confirmed, or use **Git LFS** (`git lfs track "*.pt" "*.pth"`) in a private repository.

---

## 2. Verified Experimental Results

All metrics below are extracted directly from the saved training artifacts (`results.csv` and `summary.txt`) preserved under `research/evaluation/`. No numbers have been fabricated or interpolated.

### 2.1 YOLOv8 Experiments (`research/evaluation/yolo/runs/`)

| Run Folder | Base Model | Dataset Config | Epochs | Precision (B) | Recall (B) | mAP50 (B) | mAP50-95 (B) | Status & Notes |
|---|---|---|---|---|---|---|---|---|
| `yolov8s_reid_train14__APP_best` | `yolov8s.pt` | `reid.v1i.yolov8` (MIT) | 50 / 50 | **0.8581** | **0.7440** | **0.8486** | **0.5444** | **Production `best.pt`**. Valid train/val split. |
| `yolov8s_yolo8_train__APP_best2` | `yolov8s.pt` | `yolo8/data.yaml` (`train` vs `valid`) | 80 / 80 | **0.8404** | **0.6999** | **0.7995** | **0.5387** | **Production `best2.pt`**. Valid train/val split. |
| `yolov8n_person_dataset_train3` | `yolov8n.pt` | `person_dataset` (CC BY 4.0) | 50 / 50 | 0.7120 | 0.5853 | 0.6742 | 0.4516 | Completed baseline nano run (`person_model.pt`). |
| `yolov8n_person_dataset_train` | `yolov8n.pt` | `person_dataset` (CC BY 4.0) | 28 / 50 | 0.6529 | 0.5832 | 0.6423 | 0.4076 | Early-stopped/interrupted at epoch 28. |
| `yolov8l_MOT20yaml_train15` | `yolov8l.pt` | `data_MOT20_YOLO.yaml` | 49 / 60 | 0.9938* | 0.9683* | 0.9865* | 0.9142* | **Caveat:** `data.yaml` had `val: images/train` (evaluated on training split). Do not cite as held-out test accuracy. |
| `yolov8l_MOT20yaml_train16` | `yolov8l.pt` | `data_MOT20_YOLO.yaml` | 60 / 60 | 0.9932* | 0.9649* | 0.9864* | 0.8318* | **Caveat:** Same `val: images/train` overlap as `train15`. |
| `aborted_train4` .. `aborted_train13` | `yolov8n/s` | `reid` variants | 0 | — | — | — | — | Aborted startup attempts (`args.yaml` only). |

*Other YOLO folders inspected in the original project:*
- **YOLOv9 (`yolov9/`):** Contained an upstream clone of `WongKinYiu/yolov9` and `train_custom.py` (preserved in `research/training/yolo/train_yolov9_custom.py`), but `0` saved `.pt` checkpoints and no completed `results.csv`.
- **YOLOv10 (`yolov10/`):** Contained an upstream clone of `THU-MIG/yolov10` and two inference-only preview videos (`predict/0.avi`, `predict2/0.avi`), with no custom training runs.
- **YOLO11 (`yolo11/`):** Contained `best.pt` (timestamped `2025-02-19`, Ultralytics `8.3.77`) and a template `readme.txt` with unverified provenance. Preserved as reference scripts in `research/evaluation/yolo/` and marked `UNVERIFIED`.

### 2.2 CSRNet Experiments (`research/evaluation/csrnet_shanghaitech/results/summary.txt`)

- **Model:** PyTorch `CSRNet` (`VGG16` front-end `features[:23]` + 6 dilated convolution layers with dilation rate 2 + $1 \times 1$ output layer).
- **Training Setup (`research/training/csrnet_shanghaitech/train.py`):**
  - Trained jointly on **ShanghaiTech Part A + Part B** (`ConcatDataset`), `EPOCHS = 120`, `BATCH = 8`, `LR = 5e-6`, `Adam` (`weight_decay = 1e-4`), random $384 \times 384$ crops, sum-preserving density map downsampling, density sharpening, background suppression loss, and A/B scene-density weighting.
- **Evaluation Setup (`research/evaluation/csrnet_shanghaitech/eval_csrnet.py`):**
  - Overlapping $384 \times 384$ patch inference (`stride = 192`) with 2D Gaussian window blending ($\sigma = 0.5$) across ShanghaiTech Part A + Part B test sets.
- **Verified Test Metrics (`summary.txt`):**
  - **MAE:** `35.84`
  - **RMSE:** `69.92`
  - **Under-count ratio:** `0.540`
  - **Over-count ratio:** `0.460`
  - **Recall-like ($<10\%$ error):** `0.514`

---

## 3. Inventory of Large & Restricted Files Excluded from Git

The original development workspace totaled **~87.1 GB** across `403,472` files. The following categories were intentionally excluded from the clean working copy and/or blocked via `.gitignore`:

| Excluded Category | Original Location(s) | Approx. Size | Reason for Exclusion |
|---|---|---|---|
| **Python Virtual Environments (9 venvs)** | `bitirme-backend/venv`, `bitirme_projesi/{.venv, venv, crowd_env, csrnet_env, csrnet/csrnet_env, csrnet_tf, ucf_qnrf/csrnet_tf, yolo11_env, yolo8/yolo8_env, yolov10_env, yolov9/yolov9_env}` | ~43.0 GB | Non-portable binaries & hardcoded local paths; replaced by `requirements.txt` and `requirements-lock.txt`. |
| **Generated Video & Image Outputs** | `bitirme-backend/processed/` (569 files, up to 1.17 GB per MP4), `bitirme-backend/uploads/` (164 files) | ~19.1 GB | Runtime user/test artifacts containing potentially identifiable faces and large uncompressed/transcoded media. |
| **Public & Academic Datasets** | `MOT20/` (4.74 GB), `ucf_qnrf/` (6.23 GB), `ShanghaiTech/` (3.37 GB), `crowd_counting_project/jhu_crowd_v2.0.zip` (3.08 GB), `crowdhuman.zip` (348 MB), `yolo8/{train,valid}`, `train/`, `valid/`, `data/` | ~22.5 GB | Exceeds GitHub size limits and governed by academic/dataset licenses that prohibit redistribution. |
| **Intermediate & Unused Model Checkpoints** | `runs/detect/train15/weights/*.pt` (262 MB each), `train16/weights/*.pt` (88 MB each), pretrained base weights (`yolov8l.pt`, `yolov9-c.pt`, `yolov9c.pt`), duplicate checkpoints | ~1.4 GB | Exceeds GitHub's 100 MB file limit or redundant with official Ultralytics downloads. |
| **Upstream Third-Party Clones** | `bitirme_projesi/yolov9`, `bitirme_projesi/yolov10`, `bitirme_projesi/ucf_qnrf/CSRNet-Keras` | ~0.5 GB | Unmodified third-party repositories with their own `.git` histories. |
| **Unverified Test Videos & Demo Photos** | `bitirme_projesi/test*.mp4`, `crowded*.mp4`, `bitirme foto video/*` (`58.jpg`, `100people.jpg`, `551.jpg`, `570.jpg`, `1588.jpg`, `test29.mp4`, `test33.mp4`) | ~1.1 GB | Redistribution rights and privacy clearances are unconfirmed (`58.jpg` is identical to a restricted UCF-QNRF test image). |
| **Local Runtime Assets (Copied Locally, Git-Ignored)** | `backend/models/{best.pt, best2.pt, csrnet_AB_best.pth}` (105.0 MB) and `frontend/public/stops/*.jpg` (11.3 MB) | 116.3 MB | Kept in the local working copy so the application runs out-of-the-box locally, but strictly excluded via `.gitignore` until redistribution rights and Git LFS / Release hosting are finalized. |
