import express from "express";
import cors from "cors";
import multer from "multer";
import { exec } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

// __dirname oluştur
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

// Geçici yükleme klasörü
const upload = multer({ dest: "uploads/" });

// 📌 YOLO Python scripti yolu
const YOLO_SCRIPT = path.join(__dirname, "yolo_count.py");

// -------------------------------------------------------
// 🟩 API: Resim yükle → YOLO sayım → sayıyı döndür
// -------------------------------------------------------
app.post("/api/count", upload.single("image"), (req, res) => {
  const imagePath = req.file.path; // yüklenen resmin yolu

  // Python komutu
  const pythonCmd = `python "${YOLO_SCRIPT}" "${imagePath}"`;

  exec(pythonCmd, (err, stdout, stderr) => {
    if (err) {
      console.error("Python hata:", err);
      return res.status(500).json({ error: "Python çalıştırılamadı" });
    }

    const count = parseInt(stdout.toString().trim(), 10);

    res.json({
      status: "success",
      count: isNaN(count) ? 0 : count,
    });
  });
});

// -------------------------------------------------------
app.listen(5000, () => {
  console.log("🚀 Backend çalışıyor: http://localhost:5000");
});
