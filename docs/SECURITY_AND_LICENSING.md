# Security Hardening, Google Maps Key Guide & Licensing Review

## 1. Security Vulnerabilities Addressed in Phase 3A

### 1.1 Arbitrary File Read & Directory Traversal in `GET /file/{path}` (Fixed)
- **Original Issue:** `backend/main.py` implemented `@app.get("/file/{path:path}")` by passing `path` directly to `os.path.exists(path)` and `FileResponse(path)`, allowing any caller to read arbitrary files on the host filesystem (e.g., `/file/main.py`, `/file/../.env`, or absolute Windows paths).
- **Fix Applied (`backend/main.py`):**
  1. Normalizes path separators and requires the path to begin with `processed/`.
  2. Rejects any relative subpath containing `/`, `..`, or `:` with HTTP `403 Forbidden`.
  3. Resolves the canonical path via `Path.resolve()` and enforces `candidate.relative_to(PROCESSED_DIR)`.
  4. Enforces an explicit media extension allowlist (`{.jpg, .jpeg, .png, .mp4}`).
  5. Sanitizes uploaded filenames via `_safe_filename()` before saving to `uploads/`.
- **Verification:** Covered by automated regression tests in `backend/tests/test_security_and_api.py` (`test_01_file_serving_blocks_traversal_and_source_files`), testing 11 traversal/source-file vectors and live HTTP requests.

### 1.2 Unrestricted Wildcard CORS with Credentials (Fixed)
- **Original Issue:** `CORSMiddleware` was configured with `allow_origins=["*"]` and `allow_credentials=True`.
- **Fix Applied (`backend/main.py`):** Replaced `"*"` with an explicit origin list parsed from the `CORS_ALLOWED_ORIGINS` environment variable (defaulting to `http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000` for local development) and restricted `allow_methods` to `["GET", "POST", "OPTIONS"]`.
- **Verification:** Verified both in unit tests (`test_02_cors_configuration_is_restricted`) and against a live Uvicorn HTTP server (`Origin: http://localhost:5173` receives `Access-Control-Allow-Origin: http://localhost:5173`, whereas `Origin: https://malicious.example.com` receives no `Access-Control-Allow-Origin` header).

### 1.3 Hardcoded Google Maps API Keys in Frontend Source (Fixed in Code; Key Rotation Required by Owner)
- **Original Issue:** Two distinct `AIza...` Google Maps API keys were hardcoded in `bitirme-frontend/index.html` and `bitirme-frontend/src/config.js`.
- **Fix Applied (`frontend/index.html` & `frontend/src/config.js`):**
  - Both key literals were stripped during the working-copy build (`0` occurrences of `AIza` exist anywhere in `C:\Projects\hybrid-crowd-counting`).
  - `frontend/index.html` now uses Vite HTML env substitution (`key=%VITE_GOOGLE_MAPS_API_KEY%`), and `frontend/src/config.js` reads `import.meta.env.VITE_GOOGLE_MAPS_API_KEY`, `import.meta.env.VITE_GOOGLE_MAPS_MAP_ID`, and `import.meta.env.VITE_API_BASE_URL`.
  - `.env.example` templates are provided at the repo root, `frontend/.env.example`, and `backend/.env.example`, while `.gitignore` blocks `.env` and `.env.*`.

> [!IMPORTANT]
> **Why `VITE_GOOGLE_MAPS_API_KEY` Is Not a Secret:**
> Vite statically replaces `import.meta.env.VITE_*` and `%VITE_*%` in client-side code at build time. Furthermore, the browser must send the Google Maps API key in every request to `https://maps.googleapis.com/maps/api/js`. Moving the key to `.env` keeps it out of your public Git repository, **not** out of the user's browser.
>
> **Mandatory Owner Actions in Google Cloud Console Before Going Public:**
> 1. **Rotate / Regenerate** both previously hardcoded API keys in Google Cloud Console -> *APIs & Services* -> *Credentials* (the old keys have **not** been rotated by this migration).
> 2. Under **Application restrictions**, select **Websites (HTTP referrers)** and allow only your authorized origins (e.g., `http://localhost:5173/*`, `http://127.0.0.1:5173/*`, and your production domain).
> 3. Under **API restrictions**, select **Restrict key** and enable **only** the **Maps JavaScript API**.
> 4. Configure budget/quota alerts on your Google Cloud billing account.

---

## 2. Source-Code & Dependency Licensing Review

| Component / Dependency | License | Implications for Public GitHub Release |
|---|---|---|
| **Ultralytics (`ultralytics==8.3.237`)** | **AGPL-3.0** | Ultralytics YOLOv8 is licensed under GNU AGPLv3 (unless you hold a commercial Ultralytics Enterprise license). If you publish this repository publicly as open-source software, licensing the repository under **AGPL-3.0** (or clearly documenting that the YOLOv8 integration is governed by AGPL-3.0) satisfies open-source copyleft obligations. |
| **PyTorch (`torch`, `torchvision`)** | BSD-3-Clause | Permissive; compatible with open-source distribution. |
| **FastAPI, Starlette, Uvicorn, Pydantic** | MIT / BSD-3-Clause | Permissive. |
| **OpenCV (`opencv-python`), NumPy, SciPy, Pillow** | Apache-2.0 / BSD / HPND | Permissive. |
| **React, Vite, Tailwind CSS, Chart.js, deck.gl** | MIT | Permissive. |
| **Training Datasets (Excluded from Git)** | Mixed (`MIT`, `CC BY 4.0`, Academic Non-Commercial) | `reid.v1i.yolov8` is MIT; `person_dataset` and `crowd-human-detection v2` are **CC BY 4.0** (attribution preserved in `research/configs/`). **ShanghaiTech**, **UCF-QNRF**, **MOT20**, and **JHU-CROWD++** restrict redistribution and commercial use; raw datasets and unverified media must **not** be uploaded to GitHub. |
