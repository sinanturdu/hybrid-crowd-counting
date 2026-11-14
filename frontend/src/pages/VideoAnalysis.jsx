// src/pages/VideoAnalysis.jsx
import { useState, useContext, useRef } from "react";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { MapContext } from "../context/MapContext";
import { STOPS } from "../data/stops";
import { API_BASE_URL } from "../config";

ChartJS.register(
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend
);

export default function VideoAnalysis() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [processedVideo, setProcessedVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [frameData, setFrameData] = useState([]);
  const [total, setTotal] = useState(0);
  const [average, setAverage] = useState(0);
  const [stopId, setStopId] = useState("taksim");

  // 🔥 MODEL SEÇİMİ
  const [modelName, setModelName] = useState("best");

  const { addMarker } = useContext(MapContext);

  const [liveCount, setLiveCount] = useState(0);
  const [fps, setFps] = useState(25);

  const videoRef = useRef(null);
  const chartRef = useRef(null);

  // ========================================
  // 📌 SADECE VİDEO SEÇ
  // ========================================
  const handleSelectVideo = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setVideoFile(file);
    setVideoPreview(URL.createObjectURL(file));
    setProcessedVideo(null);
    setFrameData([]);
    setLiveCount(0);
    setTotal(0);
    setAverage(0);
  };

  // ========================================
  // 🔵 ANALİZ ET (BACKEND)
  // ========================================
  const handleAnalyzeVideo = async () => {
    if (!videoFile) return;

    setLoading(true);

    const formData = new FormData();
    formData.append("video", videoFile);
    formData.append("stop", stopId);
    formData.append("model_name", modelName);

    try {
      const res = await fetch(`${API_BASE_URL}/video`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      setProcessedVideo(`${API_BASE_URL}${data.processed_video}`);
      setFrameData(data.frame_results || []);
      setTotal(data.total_count || 0);
      setAverage(data.avg_people || 0);
      setFps(data.fps || 25);

      const avg = data.avg_people || 0;
      let color = "green";
      if (avg >= 20) color = "yellow";
      if (avg >= 60) color = "red";

      if (data.lat && data.lng) {
        addMarker({
          lat: data.lat,
          lng: data.lng,
          count: Math.round(avg),
          color,
          stopName: data.stop_name || stopId,
        });
      }
    } catch (err) {
      console.error("❌ Video analiz hatası:", err);
    }

    setLoading(false);
  };

  // ========================================
  // 🟡 FPS + GRAFİK SENKRON
  // ========================================
  const handleTimeUpdate = () => {
    if (!videoRef.current || frameData.length === 0) return;
    const currentTime = videoRef.current.currentTime;
    const frameIndex = Math.floor(currentTime * fps);

    if (frameData[frameIndex]) {
      setLiveCount(frameData[frameIndex].count);

      const chart = chartRef.current?.chart || chartRef.current;
      if (chart) {
        chart.data.datasets[0].pointRadius = frameData.map((_, i) =>
          i === frameIndex ? 8 : 3
        );
        chart.data.datasets[0].pointBackgroundColor = frameData.map((_, i) =>
          i === frameIndex ? "red" : "#3b82f6"
        );
        setTimeout(() => chart.update("none"), 0);
      }
    }
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
  };

  const chartData = {
    labels: frameData.map((f) => f.frame),
    datasets: [
      {
        label: "Kişi Sayısı",
        data: frameData.map((f) => f.count),
        borderWidth: 2,
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59,130,246,0.2)",
        fill: true,
        pointRadius: frameData.map(() => 3),
      },
    ],
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">🎥 Video Analizi</h1>

      {/* Durak */}
      <div className="flex items-center gap-3">
        <label className="font-medium">Durak:</label>
        <select
          value={stopId}
          onChange={(e) => setStopId(e.target.value)}
          className="border rounded px-3 py-1"
        >
          {STOPS.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>
      </div>

      {/* Model */}
      <div className="flex items-center gap-3">
        <label className="font-medium">Model:</label>
        <select
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          className="border rounded px-3 py-1"
        >
          <option value="best">Best</option>
          <option value="best2">Best2</option>
          <option value="csrnet_ab">CSRNet AB</option>
        </select>
      </div>

      {/* Video Seç */}
      <input type="file" accept="video/*" onChange={handleSelectVideo} />

      {/* Analiz Et */}
      <button
        onClick={handleAnalyzeVideo}
        disabled={loading || !videoFile}
        className="px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? "Analiz Ediliyor..." : "Analiz Et"}
      </button>

      {videoPreview && (
        <video src={videoPreview} controls className="w-full rounded shadow" />
      )}

      {processedVideo && (
        <div className="relative">
          <div className="absolute top-5 right-5 bg-black/70 text-white px-4 py-2 rounded-xl text-xl font-bold">
            👥 {liveCount}
          </div>

          <video
            src={processedVideo}
            controls
            ref={videoRef}
            onTimeUpdate={handleTimeUpdate}
            className="w-full rounded shadow border"
          />
        </div>
      )}

      {/* 📊 Grafik */}
      {frameData.length > 0 && (
        <div className="h-64">
          <Line ref={chartRef} data={chartData} options={chartOptions} />
        </div>
      )}

      {/* 📈 ORTALAMA KİŞİ SAYISI */}
      {frameData.length > 0 && (
        <div className="text-lg font-semibold bg-gray-100 p-4 rounded">
          📌 Ortalama Kişi Sayısı: <span className="font-bold">{average}</span>
        </div>
      )}

      {/* 📋 FRAME TABLOSU */}
      {frameData.length > 0 && (
        <div className="overflow-auto max-h-96 border rounded">
          <table className="min-w-full text-sm text-left">
            <thead className="bg-gray-200 sticky top-0">
              <tr>
                <th className="px-4 py-2 border">Frame</th>
                <th className="px-4 py-2 border">Kişi Sayısı</th>
              </tr>
            </thead>
            <tbody>
              {frameData.map((f, i) => (
                <tr key={i} className="odd:bg-white even:bg-gray-50">
                  <td className="px-4 py-2 border">{f.frame}</td>
                  <td className="px-4 py-2 border">{f.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
