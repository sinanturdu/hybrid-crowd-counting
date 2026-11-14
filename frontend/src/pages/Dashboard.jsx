// src/pages/Dashboard.jsx
import { useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  registerables
} from "chart.js";

ChartJS.register(...registerables);

export default function Dashboard() {
  const [imageCount, setImageCount] = useState(37);

  const chartData = {
    labels: ["Ocak", "Şubat", "Mart", "Nisan"],
    datasets: [
      {
        label: "Tespit Edilen Kişi Sayısı",
        data: [12, 19, 8, 15],
        backgroundColor: "rgba(59, 130, 246, 0.6)",
      },
    ],
  };

  return (
    <div className="animate-fadeIn">
      <h1 className="text-2xl font-bold mb-3">Dashboard</h1>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md mb-6">
        <h2 className="text-lg font-semibold mb-4">Fotoğraf Analizi</h2>
        <p className="text-sm mb-4">{imageCount}</p>
        <Bar data={chartData} />
      </div>
    </div>
  );
}
