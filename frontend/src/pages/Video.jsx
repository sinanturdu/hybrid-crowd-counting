import { useState } from "react";
import { API_BASE_URL } from "../config";

export default function Video() {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [videoResult, setVideoResult] = useState(null);
  const [videoPlot, setVideoPlot] = useState(null);

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setSelectedVideo(URL.createObjectURL(file));
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload-video`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      setVideoResult(data.counts);
      setVideoPlot(data.plot_base64);
    } catch (err) {
      console.error("Video upload error:", err);
    }

    setLoading(false);
  };

  return (
    <div className="p-6 animate-fadeIn">
      <h1 className="text-3xl font-bold mb-6">🎥 Video Analizi</h1>

      {/* FILE INPUT */}
      <input
        type="file"
        accept="video/*"
        onChange={handleVideoUpload}
        className="mb-4"
      />

      {/* VIDEO PREVIEW */}
      {selectedVideo && (
        <video
          src={selectedVideo}
          controls
          className="rounded-lg shadow mb-4 w-full max-w-xl"
        ></video>
      )}

      {/* LOADING STATE */}
      {loading && <p className="text-blue-500">Video analiz ediliyor...</p>}

      {/* RESULT TABLE */}
      {videoResult && (
        <div className="mt-6 bg-white dark:bg-gray-800 p-4 rounded shadow">
          <h2 className="text-xl font-semibold mb-3">
            📊 Saniyelere Göre Kişi Sayısı
          </h2>

          <table className="w-full border rounded overflow-hidden">
            <thead>
              <tr className="bg-gray-200 dark:bg-gray-700">
                <th className="p-2 border">Saniye</th>
                <th className="p-2 border">Kişi Sayısı</th>
              </tr>
            </thead>
            <tbody>
              {videoResult.map((item, index) => (
                <tr key={index} className="odd:bg-gray-100 dark:odd:bg-gray-900">
                  <td className="p-2 border">{item.second}</td>
                  <td className="p-2 border">{item.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* CHART IMAGE */}
      {videoPlot && (
        <img
          src={`data:image/png;base64,${videoPlot}`}
          alt="Grafik"
          className="mt-6 rounded-lg shadow-lg"
        />
      )}
    </div>
  );
}
