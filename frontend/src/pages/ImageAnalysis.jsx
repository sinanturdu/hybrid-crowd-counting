// src/pages/ImageAnalysis.jsx
import { useState, useContext } from "react";
import { MapContext } from "../context/MapContext";
import { STOPS } from "../data/stops";
import { API_BASE_URL } from "../config";

export default function ImageAnalysis() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [count, setCount] = useState(null);
  const [stopId, setStopId] = useState("mecidiyekoy");

  // 🔥 EKLENDİ: Model seçimi
  const [modelName, setModelName] = useState("best");

  // 🔥 İşlenmiş resim
  const [processedImage, setProcessedImage] = useState(null);

  const { addMarker, addHeatPoint } = useContext(MapContext);

  // ---------------------------------------------------
  // 📌 Fotoğraf seç
  // ---------------------------------------------------
  const handleImage = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  // ---------------------------------------------------
  // 📌 FOTOĞRAFI BACKEND'E YOLLA
  // ---------------------------------------------------
  const sendToBackend = async () => {
    if (!image) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", image);
    formData.append("stop", stopId);
    formData.append("model_name", modelName); // 🔥 EKLENDİ

    try {
      const res = await fetch(`${API_BASE_URL}/upload-image`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      console.log("📸 FOTOĞRAF ANALİZ YANITI:", data);

      setCount(data.count ?? 0);

      // 🔥 İşlenmiş görsel (model fark etmeksizin)
      setProcessedImage(`${API_BASE_URL}${data.processed_image}`);

      const color =
        data.count < 20 ? "#22c55e" :
        data.count < 60 ? "#f97316" :
        "#ef4444";

      // 📌 Haritaya marker + heatmap
      if (data.lat && data.lng) {
        addMarker({
          lat: Number(data.lat),
          lng: Number(data.lng),
          count: data.count,
          color,
          stopName: stopId,
        });

        addHeatPoint(Number(data.lat), Number(data.lng), data.count);
      }
    } catch (err) {
      console.error("❌ Fotoğraf yükleme hatası:", err);
    }

    setLoading(false);
  };

  return (
    <div className="animate-fadeIn space-y-4">
      <h1 className="text-2xl font-bold mb-4">📷 Fotoğraf Analizi</h1>

      {/* Durak */}
      <div className="flex items-center gap-3">
        <label className="font-medium">Durak:</label>
        <select
          value={stopId}
          onChange={(e) => setStopId(e.target.value)}
          className="border rounded px-3 py-1 bg-white dark:bg-gray-800"
        >
          {STOPS.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>
      </div>

      {/* 🔥 MODEL SEÇİMİ (EKLENDİ) */}
      <div className="flex items-center gap-3">
        <label className="font-medium">Model:</label>
        <select
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          className="border rounded px-3 py-1 bg-white dark:bg-gray-800"
        >
          <option value="best">Best (Genel)</option>
          <option value="best2">Best2</option>
          <option value="csrnet_ab">CSRNet AB</option>
        </select>
      </div>

      {/* Fotoğraf */}
      <input type="file" accept="image/*" onChange={handleImage} />

      {preview && (
        <img
          src={preview}
          alt="preview"
          className="w-80 rounded-lg shadow-lg mb-4"
        />
      )}

      <button
        onClick={sendToBackend}
        className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-60"
        disabled={loading}
      >
        {loading ? "Analiz ediliyor..." : "Analiz Et"}
      </button>

      {count !== null && (
        <p className="mt-4 text-xl font-bold">
          👥 Tespit Edilen İnsan Sayısı: {count}
        </p>
      )}

      {/* 🔥 İşlenmiş görsel */}
      {processedImage && (
        <div className="mt-4">
          <h3 className="font-semibold mb-2">
            🟢 İşlenmiş (Kutucuklu) Görsel – {modelName}
          </h3>
          <img
            src={processedImage}
            alt="processed"
            className="w-80 rounded-lg shadow-lg border"
          />
        </div>
      )}
    </div>
  );
}
