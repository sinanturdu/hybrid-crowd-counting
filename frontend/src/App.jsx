// src/App.jsx
import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";

import Dashboard from "./pages/Dashboard";
import ImageAnalysis from "./pages/ImageAnalysis";
import VideoAnalysis from "./pages/VideoAnalysis";
import MapPage from "./pages/MapPage";

export default function App() {
  return (
    <div className="flex min-h-screen dark:bg-gray-900">
      {/* Sol menü */}
      <Sidebar />

      {/* Sağ ana alan */}
      <div className="ml-64 flex-1">
        <Navbar />

        <div className="p-6 mt-16">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/image" element={<ImageAnalysis />} />
            <Route path="/video" element={<VideoAnalysis />} />
            <Route path="/map" element={<MapPage />} />
          </Routes>
        </div>
      </div>
    </div>
  );
}
