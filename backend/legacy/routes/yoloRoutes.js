
// bitirme-backend/routes/yoloRoutes.js

import express from "express";
import { spawn } from "child_process";

const router = express.Router();

// YOLO insan sayma endpoint
router.post("/count", async (req, res) => {
  try {
    const { imagePath } = req.body;

    if (!imagePath) {
      return res.status(400).json({ error: "imagePath eksik" });
    }

    const python = spawn("python", ["yolo_count.py", imagePath]);

    let output = "";

    python.stdout.on("data", (data) => {
      output += data.toString();
    });

    python.stderr.on("data", (data) => {
      console.error("YOLO STDERR:", data.toString());
    });

    python.on("close", () => {
      const count = parseInt(output.trim(), 10);
      res.json({ count });
    });
  } catch (err) {
    console.error("YOLO hata:", err);
    res.status(500).json({ error: "YOLO çalıştırılamadı" });
  }
});

export default router;
