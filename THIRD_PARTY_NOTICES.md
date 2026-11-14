# Third-Party Notices, Code Provenance & Academic Citations

This document records the provenance, copyright notices, license terms, and unresolved licensing boundaries for third-party dependencies, adapted research routines, academic architectures, and training datasets referenced by the **Hybrid Crowd Counting & Public Transit Density Analysis** repository.

---

## 1. License Scope & Author Copyright Notice

```text
Hybrid Crowd Counting & Public Transit Density Analysis (Original Project Code)
Copyright (C) 2025 Sinan Türdü
SPDX-License-Identifier: AGPL-3.0-only

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program (see LICENSE). If not, see <https://www.gnu.org/licenses/>.
```

The root [`LICENSE`](LICENSE) contains the **unmodified verbatim text** of the **GNU Affero General Public License v3.0** (`Version 3, 19 November 2007`, Free Software Foundation, Inc.) and applies **exclusively** to the original source code, web application logic, and documentation written by **Sinan Türdü**:

- **Original Backend Code:** [`backend/main.py`](backend/main.py), [`backend/models/csrnet.py`](backend/models/csrnet.py), [`backend/tests/test_security_and_api.py`](backend/tests/test_security_and_api.py), and legacy prototypes in [`backend/legacy/`](backend/legacy/).
- **Original Frontend Code:** All React 19 / Vite 7 components, pages, state context, and configuration files under [`frontend/src/`](frontend/src/).
- **Original Research & Training Code:** Custom PyTorch training loops, sum-preserving density resizing, background suppression and A/B scene-aware loss functions, $384 \times 384$ overlapping Gaussian patch inference, and evaluation pipelines in [`research/`](research/), **except** for the specific third-party dataset metadata files and adapted research routines identified below.

Adding the root `LICENSE` file **does not** relicense third-party libraries, adapted external snippets, academic datasets, or unbundled model checkpoints that the author does not own.

---

## 2. Core Runtime & Research Dependencies

| Package / Component | Verified Version | License | Role in Project |
|---|---|---|---|
| **Ultralytics YOLOv8** (`ultralytics`) | `8.3.237` | **GNU AGPL-3.0** (or commercial Ultralytics Enterprise License) | Object detection model loading (`YOLO`), training, and person detection inference in `backend/main.py` and `research/`. |
| **PyTorch & Torchvision** (`torch`, `torchvision`) | `2.5.1+cu121`, `0.20.1+cu121` | **BSD-3-Clause** | Deep learning runtime, CSRNet `nn.Module` implementation, and VGG16 front-end backbone (`models.vgg16`). |
| **FastAPI, Starlette, Uvicorn, Pydantic** | `0.121.3`, `0.38.0` | **MIT / BSD-3-Clause** | REST API server, multipart file handling, and OpenAPI schema generation. |
| **OpenCV (`opencv-python`), NumPy, SciPy, Pillow, h5py** | `4.12.0.88`, `2.1.2`, `1.15.3`, `12.0.0` | **Apache-2.0 / BSD-3-Clause / HPND** | Frame decoding, colormap rendering, KDTree spatial queries, Gaussian filtering, and HDF5/MAT parsing. |
| **FFmpeg** (external system binary) | System / `libx264` | **GPLv2+ / LGPLv2.1+** (invoked as a separate CLI subprocess) | Transcoding temporary OpenCV `.avi` outputs into browser-compatible H.264 `.mp4` streams. |
| **React, React Router, Vite, Tailwind CSS, Chart.js, deck.gl, Lucide** | React `19.2.0`, Vite `7.2.4`, Chart.js `4.5.1`, deck.gl `9.2.5` | **MIT / Apache-2.0** | Frontend UI, time-series synchronization charts, and geospatial `HeatmapLayer` rendering. |

---

## 3. Adapted Research Routines & Preserved Third-Party Metadata

### 3.1 Geometry-Adaptive KDTree Gaussian Density Map Routine
- **File:** [`research/preprocessing/csrnet_shanghaitech/generate_h5_from_mat.py`](research/preprocessing/csrnet_shanghaitech/generate_h5_from_mat.py) (`gaussian_filter_density`, lines 21–42)
- **Provenance:** The `gaussian_filter_density(gt)` function adapts the standard geometry-adaptive Gaussian kernel algorithm ($k=4$ nearest neighbors via `scipy.spatial.KDTree`, $\sigma_i = 0.1 \sum_{j=1}^{3} d_{ij}$) introduced by Zhang et al. (*CVPR 2016*) and popularized in open-source Python by [`leeyeehoo/CSRNet-pytorch`](https://github.com/leeyeehoo/CSRNet-pytorch) and [`vlad3996/computing-density-maps`](https://github.com/vlad3996/computing-density-maps). The surrounding parallel `.h5` generation pipeline (`generate_density_map`, `collect_image_paths`, `joblib.Parallel`) was written for this project.
- **License Note:** Because the upstream [`leeyeehoo/CSRNet-pytorch`](https://github.com/leeyeehoo/CSRNet-pytorch) repository did not publish a formal `LICENSE` file in its root, the adapted 22-line `gaussian_filter_density` routine is explicitly attributed here to its upstream authors rather than claimed as original work.

### 3.2 Preserved Dataset README & Attribution Files in `research/configs/`
The following metadata files were exported alongside external datasets and remain governed by their respective upstream terms:
- [`research/configs/dataset_reid_README.dataset.txt`](research/configs/dataset_reid_README.dataset.txt) & [`dataset_reid_README.roboflow.txt`](research/configs/dataset_reid_README.roboflow.txt) — Dataset: `reid-pgi5t` (`https://universe.roboflow.com/bq/reid-pgi5t/dataset/1`), **License: MIT**.
- [`research/configs/dataset_crowdhuman_v2_README.dataset.txt`](research/configs/dataset_crowdhuman_v2_README.dataset.txt) & [`dataset_crowdhuman_v2_README.roboflow.txt`](research/configs/dataset_crowdhuman_v2_README.roboflow.txt) — Dataset: `crowd-human-detection v2` (`https://universe.roboflow.com/digitalna-dalmacija-7qxuq/crowd-human-detection/dataset/2`), **License: CC BY 4.0** (attribution required).
- [`research/configs/dataset_person_dataset_data.yaml`](research/configs/dataset_person_dataset_data.yaml) — References `person-pi7xb` (`https://universe.roboflow.com/dng-cjryd/person-pi7xb/dataset/1`), **License: CC BY 4.0** (attribution required).
- [`research/configs/ucf_qnrf_readme.txt`](research/configs/ucf_qnrf_readme.txt) — Official citation notice for the **UCF-QNRF** dataset (Idrees et al., *ECCV 2018*).
- [`research/configs/yolo11_readme_UNVERIFIED.txt`](research/configs/yolo11_readme_UNVERIFIED.txt) — Archived documentation from an external/unverified YOLO11 experiment; not used by the runtime application.

---

## 4. Academic Architecture & Dataset Citations

No raw dataset images, videos, or point/bounding-box annotation archives are distributed in this repository. If you use the architectures or reproduce the experiments in [`research/`](research/), please cite the original publications:

```bibtex
@inproceedings{li2018csrnet,
  title={CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes},
  author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1091--1100},
  year={2018}
}

@inproceedings{zhang2016single,
  title={Single-Image Crowd Counting via Multi-Column Convolutional Neural Network},
  author={Zhang, Yingying and Zhou, Desen and Chen, Siqin and Gao, Shenghua and Ma, Yi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={589--597},
  year={2016}
}

@inproceedings{idrees2018composition,
  title={Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds},
  author={Idrees, Haroon and Tayyab, Muhmmad and Athrey, Kishan and Zhang, Dong and Al-Maadeed, Somaya and Rajpoot, Nasir and Shah, Mubarak},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={532--546},
  year={2018}
}

@article{dendorfer2020mot20,
  title={MOT20: A benchmark for multi object tracking in crowded scenes},
  author={Dendorfer, Patrick and Rezatofighi, Hamid and Milan, Anton and Shi, Javen and Cremers, Daniel and Reid, Ian and Roth, Stefan and Schindler, Konrad and Leal-Taix{\'e}, Laura},
  journal={arXiv preprint arXiv:2003.09003},
  year={2020}
}
```

---

## 5. Model Checkpoints & Unresolved Licensing Boundaries (Option C — Local-Only)

All trained model weights (`best.pt`, `best2.pt`, `csrnet_AB_best.pth`) are **excluded from this repository** (`.gitignore`) and are **not** publicly redistributed:

1. **`csrnet_AB_best.pth` (`65.1 MB`):** Trained on the **ShanghaiTech Part A + Part B** dataset (Zhang et al., CVPR 2016) initialized from Torchvision ImageNet VGG16 weights. Because ShanghaiTech is distributed for **non-commercial academic research only** and does not grant an explicit open-source redistribution license for derivative weights, this checkpoint is kept local-only.
2. **`best.pt` (`22.5 MB`) & `best2.pt` (`22.5 MB`):** Fine-tuned from Ultralytics `yolov8s.pt` (**AGPL-3.0**) on `reid.v1i.yolov8` (**MIT**) and local person-detection splits. While AGPL-3.0 Section 1 defines "Corresponding Source" for software programs, legal interpretations differ on whether distributing a software application requires distributing fine-tuned neural network weight files, or whether fine-tuned weights remain constrained by upstream base-model and dataset terms. By publishing a **code-only repository** (Option C) with full training scripts (`research/training/`) and clean `503 Service Unavailable` runtime handling when weights are absent, the repository avoids any unauthorized redistribution of third-party dataset derivatives.
