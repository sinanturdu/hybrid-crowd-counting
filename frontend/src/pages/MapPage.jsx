// src/pages/MapPage.jsx
import { useEffect, useRef, useContext, useState } from "react";
import { MapContext } from "../context/MapContext";
import { GOOGLE_MAPS_MAP_ID } from "../config";
import { GoogleMapsOverlay } from "@deck.gl/google-maps";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";

export default function MapPage() {
  const mapRef = useRef(null);
  const heatmapRef = useRef(null);

  const { registerMap, heatmapPoints, refreshHeatmap } = useContext(MapContext);

  const [heatmapEnabled, setHeatmapEnabled] = useState(false);

  const mapInstanceRef = useRef(null);

  async function initMap() {
    console.log("🗺️ Harita yükleniyor...");
    if (!window.google?.maps?.importLibrary) {
      console.warn("Google Maps JS API is not loaded. Set VITE_GOOGLE_MAPS_API_KEY in .env.");
      return;
    }

    const { Map } = await window.google.maps.importLibrary("maps");
    await window.google.maps.importLibrary("marker");

    const map = new Map(mapRef.current, {
      center: { lat: 41.015137, lng: 28.97953 },
      zoom: 12,
      gestureHandling: "greedy",
      mapId: GOOGLE_MAPS_MAP_ID,
    });

    registerMap(map);
    mapInstanceRef.current = map;

    heatmapRef.current = new GoogleMapsOverlay({
      layers: [],
    });

    heatmapRef.current.setMap(heatmapEnabled ? map : null);

    await refreshHeatmap();

    console.log("🔥 Heatmap oluşturuldu!");
  }

  useEffect(() => {
    if (!heatmapRef.current) return;

    const heatmapLayer = new HeatmapLayer({
      id: "heatmap-layer",
      data: heatmapPoints,

      getPosition: (p) => [p.lng, p.lat],
      getWeight: (p) => p.weight,

      radiusPixels: 55,
      intensity: 1,
      threshold: 0.05,

      colorRange: [
        [0, 255, 0, 0],
        [0, 255, 0, 255],
        [255, 255, 0, 255],
        [255, 165, 0, 255],
        [255, 80, 0, 255],
        [255, 0, 0, 255],
        [150, 0, 0, 255],
      ],
    });

    heatmapRef.current.setProps({
      layers: [heatmapLayer],
    });
  }, [heatmapPoints]);

  useEffect(() => {
    if (!heatmapRef.current || !mapInstanceRef.current) return;

    heatmapRef.current.setMap(
      heatmapEnabled ? mapInstanceRef.current : null
    );
  }, [heatmapEnabled]);

  useEffect(() => {
    initMap();
  }, []);

  return (
    <div className="animate-fadeIn relative">
      <button
        onClick={() => setHeatmapEnabled((prev) => !prev)}
        className="absolute z-10 top-4 right-4 px-4 py-2 bg-white border rounded shadow hover:bg-gray-100"
      >
        {heatmapEnabled ? "Heatmap Kapat" : "Heatmap Aç"}
      </button>

      <div
        ref={mapRef}
        id="map"
        className="w-full h-[650px] rounded-lg shadow-lg border"
      />
    </div>
  );
}