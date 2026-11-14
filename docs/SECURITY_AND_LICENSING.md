# Security Hardening, Google Maps Key Guide & Licensing Review

## 1. Security Vulnerabilities Addressed

### 1.1 Arbitrary File Read & Directory Traversal in `GET /file/{path}` (Fixed)
- **Original Issue:** `backend/main.py` previously passed `path` directly to `os.path.exists(path)` and `FileResponse(path)`, allowing arbitrary file reads outside the media output folder.
- **Fix Applied ([`backend/main.py`](../backend/main.py)):**
  1. Normalizes path separators and requires the requested path to begin with `processed/`.
  2. Rejects any subpath containing `/`, `..`, or `:` with HTTP `403 Forbidden`.
  3. Resolves the canonical filesystem path via `Path.resolve()` and enforces `candidate.relative_to(PROCESSED_DIR)`.
  4. Enforces an explicit media extension allowlist (`{.jpg, .jpeg, .png, .mp4}`).
  5. Sanitizes uploaded filenames via `_safe_filename()` before writing to `backend/uploads/`.
- **Verification:** Covered by automated regression tests in [`backend/tests/test_security_and_api.py`](../backend/tests/test_security_and_api.py) (`test_01_file_serving_blocks_traversal_and_source_files`).

### 1.2 Unrestricted Wildcard CORS with Credentials (Fixed)
- **Original Issue:** `CORSMiddleware` was configured with `allow_origins=["*"]` and `allow_credentials=True`.
- **Fix Applied ([`backend/main.py`](../backend/main.py)):** Replaced `"*"` with an explicit origin allowlist parsed from `CORS_ALLOWED_ORIGINS` (defaulting to `http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000` for local development) and restricted `allow_methods` to `["GET", "POST", "OPTIONS"]`.
- **Verification:** Verified by `test_02_cors_configuration_is_restricted` in [`backend/tests/test_security_and_api.py`](../backend/tests/test_security_and_api.py).

### 1.3 Safe DOM InfoWindow Construction & Missing-Asset Handling (Fixed)
- **Original Issue:** `frontend/src/context/MapContext.jsx` interpolated stop metadata into raw HTML strings for Google Maps `InfoWindow.setContent(...)`.
- **Fix Applied ([`frontend/src/context/MapContext.jsx`](../frontend/src/context/MapContext.jsx)):** `createModernPopup` now constructs a DOM `HTMLElement` via `document.createElement`, assigns text fields via `.textContent` (preventing HTML injection / XSS), and attaches a standard DOM `error` event listener (`imgEl.addEventListener("error", ...)`) so missing local-only `/stops/*.jpg` photos hide gracefully on a clean clone without inline JavaScript.

### 1.4 Hardcoded Google Maps API Keys Removed from Code (Owner Key Rotation Required)
- **Original Issue:** Two Google Maps API keys (`AIza...`) were hardcoded in the original local frontend prototype.
- **Fix Applied ([`frontend/index.html`](../frontend/index.html) & [`frontend/src/config.js`](../frontend/src/config.js)):**
  - Removed all key literals before initializing Git (`0` occurrences exist in the repository or commit history).
  - Configured Vite environment substitution (`%VITE_GOOGLE_MAPS_API_KEY%` and `import.meta.env.VITE_GOOGLE_MAPS_API_KEY`) with placeholder-only `.env.example` templates.

> [!IMPORTANT]
> **Mandatory Owner Actions in Google Cloud Console Before Switching Repository Visibility to PUBLIC:**
>
> Because Vite statically embeds `VITE_*` variables into client-side JavaScript bundles, moving a browser key to `.env` keeps it out of Git commits, **not** out of the user's browser network tab. In addition, the two keys that existed in the original local development folders have **not** been modified by this migration.
>
> **Exact Steps to Perform in [Google Cloud Console](https://console.cloud.google.com/apis/credentials):**
> 1. Navigate to **APIs & Services -> Credentials**.
> 2. **Revoke or Regenerate** both previously hardcoded API keys from the original local project so they cannot be used if ever exposed elsewhere.
> 3. Create or edit your active browser key (`VITE_GOOGLE_MAPS_API_KEY`) and under **Application restrictions**, select **Websites (HTTP referrers)** and add only your authorized origins:
>    - `http://localhost:5173/*`
>    - `http://127.0.0.1:5173/*`
>    - Your production domain (if deployed).
> 4. Under **API restrictions**, select **Restrict key** and enable **only** the **Maps JavaScript API**.
> 5. Configure a monthly budget alert under **Billing -> Budgets & alerts**.

---

## 2. Licensing & Redistribution Boundaries

### 2.1 Repository License (`GNU AGPL-3.0` / `AGPL-3.0-only`)
- Original code written for this graduation project in `backend/`, `frontend/`, and `research/` is licensed under **version 3 of the GNU Affero General Public License only** (`SPDX-License-Identifier: AGPL-3.0-only`, [`LICENSE`](../LICENSE)), ensuring compatibility with the core **`ultralytics==8.3.237`** dependency (which is distributed by Ultralytics Inc. under AGPL-3.0 for open-source users).
- Full provenance, third-party attributions, and academic citations are documented in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

### 2.2 Unresolved Third-Party & Model Weight Licensing Considerations
Adding a root `LICENSE` file **does not** automatically resolve third-party dataset or model checkpoint restrictions:
1. **Unbundled Model Checkpoints (Option C — Local-Only):**
   - `best.pt`, `best2.pt`, and `csrnet_AB_best.pth` are **excluded from Git** and are **not** publicly distributed.
   - `csrnet_AB_best.pth` was trained on **ShanghaiTech Part A + Part B** (Zhang et al., CVPR 2016), which restricts use to non-commercial academic research and does not grant an explicit open-source license for redistributing derivative weights. Simply labeling a weight file "Non-Commercial Only" does not automatically grant legal redistribution rights without upstream dataset owner permission.
   - For `best.pt` and `best2.pt` (fine-tuned from Ultralytics `yolov8s.pt`), keeping weights local-only while publishing the complete application source code and training scripts (`research/training/`) satisfies AGPL-3.0 source-code transparency for the software without redistributing unverified binary model artifacts.
2. **Clean-Room Replacement of Geometry-Adaptive Gaussian Density Routine:**
   - As documented in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md), an earlier 22-line `gaussian_filter_density` routine in [`research/preprocessing/csrnet_shanghaitech/generate_h5_from_mat.py`](../research/preprocessing/csrnet_shanghaitech/generate_h5_from_mat.py) that had been adapted from an unlicensed community repository (`leeyeehoo/CSRNet-pytorch`) was replaced with an original clean-room implementation (`build_geometry_adaptive_density`) derived directly from the published mathematical formula in Zhang et al. (*CVPR 2016*, Eq. 4) and Li et al. (*CVPR 2018*). No unlicensed third-party code remains in `research/`.
3. **Excluded Datasets & Local Stop Photographs:**
   - Raw datasets (**ShanghaiTech**, **UCF-QNRF**, **MOT20**, **CrowdHuman**, and Roboflow exports) and the 15 local transit stop photographs (`frontend/public/stops/*.jpg`) remain strictly excluded from the repository via [`.gitignore`](../.gitignore).
