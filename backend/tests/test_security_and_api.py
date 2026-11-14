"""
Security & integration test suite for the Hybrid Crowd Counting FastAPI backend.

Runs safely inside the isolated repository working copy.
Verifies:
  1. Arbitrary file read / directory traversal protection on GET /file/{path}
  2. Restricted CORS policy (allowed local dev origin vs disallowed external origin)
  3. Stop density endpoints (/api/all-densities, /api/bus-stop/{id}/density, /api/reset-densities)
  4. Image & video inference endpoints across all three runtime models (best, best2, csrnet_ab)
     using synthetic in-memory media generated on the fly.
  5. Clean-clone behavior when local-only model weights are absent (503 Service Unavailable).
"""
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import cv2
import numpy as np
from PIL import Image
from starlette.routing import Match

import main as backend_main


def _find_endpoint(path: str, method: str):
    scope = {"type": "http", "path": path, "method": method.upper(), "path_params": {}}
    for route in backend_main.app.routes:
        m, child_scope = route.matches(scope)
        if m == Match.FULL:
            return route.endpoint, child_scope.get("path_params", {})
    raise AssertionError(f"No route matched {method} {path}")


class BackendSecurityAndIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_01_file_serving_blocks_traversal_and_source_files(self):
        """Regression test for CVE-like arbitrary file read in /file/{path}."""
        # Create a legitimate processed image file first
        valid_file = backend_main.PROCESSED_DIR / "annotated_test_security.jpg"
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(valid_file), img)

        try:
            # 1) Legitimate processed file succeeds
            resp = await backend_main.get_file("processed/annotated_test_security.jpg")
            self.assertEqual(Path(resp.path).resolve(), valid_file.resolve())

            # 2) Attempts to read source code, parent dirs, uploads, or absolute paths MUST raise 403
            forbidden_paths = [
                "main.py",
                "../main.py",
                "processed/../main.py",
                "processed/..\\main.py",
                "processed/../../backend/main.py",
                "uploads/some_file.jpg",
                "models/best.pt",
                "processed/models/best.pt",
                str(backend_main.BASE_DIR / "main.py"),
                r"C:\Windows\win.ini",
                "processed/not_allowed.txt",
            ]
            for bad in forbidden_paths:
                with self.subTest(bad=bad):
                    with self.assertRaises(backend_main.HTTPException) as ctx:
                        await backend_main.get_file(bad)
                    self.assertEqual(
                        ctx.exception.status_code,
                        403,
                        f"Expected 403 for {bad}, got {ctx.exception.status_code}",
                    )

            # 3) Non-existent file inside processed/ with valid extension raises 404
            with self.assertRaises(backend_main.HTTPException) as ctx404:
                await backend_main.get_file("processed/does_not_exist_12345.jpg")
            self.assertEqual(ctx404.exception.status_code, 404)
        finally:
            if valid_file.exists():
                valid_file.unlink()

    async def test_02_cors_configuration_is_restricted(self):
        """Verify wildcard '*' CORS was replaced with explicit origins."""
        self.assertNotIn("*", backend_main.ALLOWED_ORIGINS)
        self.assertIn("http://localhost:5173", backend_main.ALLOWED_ORIGINS)
        self.assertIn("http://127.0.0.1:5173", backend_main.ALLOWED_ORIGINS)

    async def test_03_density_state_endpoints(self):
        """Verify reset, update, single-stop, and all-densities contracts."""
        res_reset = await backend_main.reset_densities()
        self.assertEqual(res_reset, {"success": True})

        all_d = await backend_main.get_all_densities()
        self.assertEqual(len(all_d), 15)
        self.assertTrue(all(item["people"] == 0 for item in all_d))

        backend_main.update_density("taksim", 27)
        single = await backend_main.get_bus_stop_density("taksim")
        self.assertEqual(
            single,
            {"stop_id": "taksim", "people": 27, "density": "Yüksek"},
        )
        await backend_main.reset_densities()

    async def test_04_image_inference_all_three_models(self):
        """Run /upload-image across best, best2, and csrnet_ab with a synthetic RGB image."""
        pil_img = Image.fromarray(np.full((384, 384, 3), 128, dtype=np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        payload = buf.getvalue()

        for model_key in ("best", "best2", "csrnet_ab"):
            with self.subTest(model_name=model_key):
                upload = backend_main.UploadFile(
                    filename="synthetic_test.jpg",
                    file=io.BytesIO(payload),
                )
                out = await backend_main.upload_image(
                    file=upload, stop="mecidiyekoy", model_name=model_key
                )
                self.assertIn("count", out)
                self.assertIsInstance(out["count"], int)
                self.assertGreaterEqual(out["count"], 0)
                self.assertEqual(out["model_used"], model_key)
                self.assertTrue(out["processed_image"].startswith("/file/processed/annotated_"))

                # Verify the generated file can be served through get_file() and then clean up
                rel_From_url = out["processed_image"][len("/file/"):]
                file_resp = await backend_main.get_file(rel_From_url)
                served_path = Path(file_resp.path)
                self.assertTrue(served_path.is_file())
                served_path.unlink(missing_ok=True)

        # Clean up synthetic uploads created during this test
        for up in backend_main.UPLOADS_DIR.glob("*_synthetic_test.jpg"):
            up.unlink(missing_ok=True)

    async def test_05_video_inference_yolo_and_csrnet(self):
        """Run /video with a tiny 6-frame 384x384 synthetic MP4 for YOLO and CSRNet."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(tmp_path), fourcc, 10.0, (384, 384))
            for i in range(6):
                frame = np.full((384, 384, 3), 100 + i * 10, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            video_bytes = tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

        for model_key in ("best", "csrnet_ab"):
            with self.subTest(model_name=model_key):
                upload = backend_main.UploadFile(
                    filename="synthetic_clip.mp4",
                    file=io.BytesIO(video_bytes),
                )
                out = await backend_main.analyze_video(
                    video=upload, stop="taksim", model_name=model_key
                )
                self.assertEqual(out["model_used"], model_key)
                self.assertIsInstance(out["total_count"], int)
                self.assertIsInstance(out["avg_people"], int)
                self.assertGreater(len(out["frame_results"]), 0)
                self.assertTrue(out["processed_video"].startswith("/file/processed/processed_"))

                rel_from_url = out["processed_video"][len("/file/"):]
                file_resp = await backend_main.get_file(rel_from_url)
                served_path = Path(file_resp.path)
                self.assertTrue(served_path.is_file())
                self.assertGreater(served_path.stat().st_size, 0)
                served_path.unlink(missing_ok=True)

        for up in backend_main.UPLOADS_DIR.glob("*_synthetic_clip.mp4"):
            up.unlink(missing_ok=True)
        await backend_main.reset_densities()

    async def test_06_clean_clone_missing_weights_returns_503(self):
        """Simulate a fresh git clone without local model weights and verify 503 response."""
        saved_cache = dict(backend_main._MODEL_CACHE)
        saved_csrnet = backend_main._CSRNET_MODEL
        saved_best = backend_main.MODEL_REGISTRY["best"]
        saved_csr_path = backend_main.MODEL_REGISTRY["csrnet_ab"]
        try:
            backend_main._MODEL_CACHE.clear()
            backend_main._CSRNET_MODEL = None
            backend_main.MODEL_REGISTRY["best"] = str(backend_main.MODELS_DIR / "nonexistent_best.pt")
            backend_main.MODEL_REGISTRY["csrnet_ab"] = str(backend_main.MODELS_DIR / "nonexistent_csrnet.pth")

            with self.assertRaises(backend_main.HTTPException) as ctx_yolo:
                backend_main.get_model("best")
            self.assertEqual(ctx_yolo.exception.status_code, 503)
            self.assertIn("backend/models/", ctx_yolo.exception.detail)

            with self.assertRaises(backend_main.HTTPException) as ctx_csr:
                backend_main.get_csrnet_model()
            self.assertEqual(ctx_csr.exception.status_code, 503)
            self.assertIn("backend/models/", ctx_csr.exception.detail)
        finally:
            backend_main.MODEL_REGISTRY["best"] = saved_best
            backend_main.MODEL_REGISTRY["csrnet_ab"] = saved_csr_path
            backend_main._MODEL_CACHE.update(saved_cache)
            backend_main._CSRNET_MODEL = saved_csrnet


if __name__ == "__main__":
    unittest.main(verbosity=2)
