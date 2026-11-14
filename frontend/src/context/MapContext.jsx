// src/context/MapContext.jsx
import { createContext, useState, useEffect, useRef } from "react";
import { STOPS } from "../data/stops";
import { API_BASE_URL } from "../config";

export const MapContext = createContext();

export default function MapProvider({ children }) {
  const [map, setMap] = useState(null);
  const infoWindowRef = useRef(null);

  function getInfoWindow() {
    if (!infoWindowRef.current && typeof window !== "undefined" && window.google?.maps?.InfoWindow) {
      infoWindowRef.current = new window.google.maps.InfoWindow();
    }
    return infoWindowRef.current;
  }

  const [densityMarkers, setDensityMarkers] = useState([]);
  const [heatmapPoints, setHeatmapPoints] = useState([]);

  // ======================================================
  // RESET BACKEND DENSITIES
  // ======================================================
  useEffect(() => {
    (async () => {
      try {
        await fetch(`${API_BASE_URL}/api/reset-densities`, {
          method: "POST",
        });
      } catch {
        // ignore offline backend on initial mount
      }
    })();
  }, []);

  // ======================================================
  // LOCAL DENSITY LABEL
  // ======================================================
  function calcLocalDensity(count) {
    const c = Number(count || 0);
    if (c <= 15) return "Düşük";
    if (c <= 35) return "Orta";
    return "Yüksek";
  }

  // ======================================================
  // POPUP (PHOTO + TEXT — DOM NODE WITH SAFE ERROR HANDLER)
  // ======================================================
  function createModernPopup({ stopName, density, people, imageUrl }) {
    const safeImage = imageUrl || "/stops/default_stop.jpg";

    const container = document.createElement("div");
    Object.assign(container.style, {
      padding: "14px 18px",
      borderRadius: "14px",
      background: "rgba(255, 255, 255, 0.88)",
      backdropFilter: "blur(8px)",
      boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
      fontFamily: "'Inter', sans-serif",
      minWidth: "220px",
    });

    const titleEl = document.createElement("div");
    Object.assign(titleEl.style, {
      fontSize: "18px",
      fontWeight: "700",
      color: "#111",
    });
    titleEl.textContent = String(stopName ?? "");
    container.appendChild(titleEl);

    const imgEl = document.createElement("img");
    imgEl.src = safeImage;
    imgEl.alt = String(stopName ?? "Stop photo");
    Object.assign(imgEl.style, {
      width: "100%",
      marginTop: "10px",
      borderRadius: "10px",
      boxShadow: "0 3px 8px rgba(0,0,0,0.2)",
    });
    imgEl.addEventListener("error", () => {
      imgEl.style.display = "none";
    });
    container.appendChild(imgEl);

    const densityRow = document.createElement("div");
    Object.assign(densityRow.style, {
      marginTop: "12px",
      fontSize: "14px",
      color: "#555",
    });
    const densityLabel = document.createElement("b");
    densityLabel.textContent = "Yoğunluk: ";
    densityRow.appendChild(densityLabel);

    const badgeEl = document.createElement("span");
    Object.assign(badgeEl.style, {
      padding: "2px 8px",
      borderRadius: "6px",
      fontWeight: "600",
      background:
        density === "Yüksek"
          ? "#fee2e2"
          : density === "Orta"
          ? "#fef9c3"
          : "#dcfce7",
      color:
        density === "Yüksek"
          ? "#b91c1c"
          : density === "Orta"
          ? "#a16207"
          : "#166534",
    });
    badgeEl.textContent = String(density ?? "Bilgi yok");
    densityRow.appendChild(badgeEl);
    container.appendChild(densityRow);

    const peopleRow = document.createElement("div");
    Object.assign(peopleRow.style, {
      marginTop: "8px",
      fontSize: "14px",
      color: "#333",
      display: "flex",
      alignItems: "center",
      gap: "6px",
    });
    const iconSpan = document.createElement("span");
    iconSpan.style.fontSize = "16px";
    iconSpan.textContent = "👥";
    const countBold = document.createElement("b");
    countBold.textContent = String(people ?? 0);
    const suffixText = document.createTextNode(" kişi");
    peopleRow.append(iconSpan, countBold, suffixText);
    container.appendChild(peopleRow);

    return container;
  }

  // ======================================================
  // GRADIENT COLOR (HER SAYI FARKLI TON – GELİŞTİRİLMİŞ)
  // ======================================================
  function getColorByDensity(count) {
    const max = 60;
    const ratio = Math.min(count / max, 1);

    // 🔥 Renk geçişini daha belirgin yaptık
    const enhancedRatio = Math.pow(ratio, 1.2);
    const greenBoost = Math.pow(1 - ratio, 0.85);

    const red = Math.floor(255 * enhancedRatio);
    const green = Math.floor(255 * greenBoost);

    return `rgb(${red}, ${green}, 0)`;
  }

  // ======================================================
  // MAP REGISTER
  // ======================================================
  function registerMap(m) {
    setMap(m);
  }

  // ======================================================
  // GREY STOP MARKERS
  // ======================================================
  function createGreyPin() {
    return new window.google.maps.marker.PinElement({
      background: "#9ca3af",
      borderColor: "#6b7280",
      glyphColor: "white",
    });
  }

  function loadStopMarkers() {
    if (!map || !window.google?.maps?.marker) return;

    STOPS.forEach((stop) => {
      const pin = createGreyPin();

      const marker = new window.google.maps.marker.AdvancedMarkerElement({
        map,
        position: { lat: stop.lat, lng: stop.lng },
        content: pin.element,
        title: stop.name,
      });

      marker.addListener("click", async () => {
        let density = "Bilgi yok";
        let people = 0;

        try {
          const res = await fetch(
            `${API_BASE_URL}/api/bus-stop/${stop.id}/density`
          );
          const data = await res.json();
          density = data?.density ?? "Bilgi yok";
          people = Math.round(data?.people ?? 0);
        } catch {
          // keep fallback label when backend is unreachable
        }

        const infoWindow = getInfoWindow();
        if (!infoWindow) return;

        infoWindow.setContent(
          createModernPopup({
            stopName: stop.name,
            density,
            people,
            imageUrl: stop.photo, // ✔️ STOP.PHOTO
          })
        );

        infoWindow.open({ map, anchor: marker });
      });
    });
  }

  // ======================================================
  // ANALYSIS MARKER (FOTO KAYBINI ENGELLEME – FIX)
  // ======================================================
  function drawDensityMarker({ lat, lng, count, stopName, imageUrl }) {
    if (!map || !window.google?.maps?.marker) return;
    const safeCount = Math.round(Number(count || 0));
    const autoColor = getColorByDensity(safeCount);

    const pin = new window.google.maps.marker.PinElement({
      background: autoColor,
      borderColor: "#000",
      glyph: safeCount.toString(),
      glyphColor: "white",
    });

    const marker = new window.google.maps.marker.AdvancedMarkerElement({
      map,
      position: { lat, lng },
      content: pin.element,
      title: stopName,
    });

    marker.addListener("click", () => {
      const stopMatch = STOPS.find(
        (s) => s.id === stopName || s.name === stopName
      );

      const finalImage =
        imageUrl || stopMatch?.photo || "/stops/default_stop.jpg";

      const infoWindow = getInfoWindow();
      if (!infoWindow) return;

      infoWindow.setContent(
        createModernPopup({
          stopName,
          density: calcLocalDensity(safeCount),
          people: safeCount,
          imageUrl: finalImage,
        })
      );

      infoWindow.open({ map, anchor: marker });
    });
  }

  // ======================================================
  // addMarker
  // ======================================================
  function addMarker({ lat, lng, count, stopName, imageUrl }) {
    const entry = {
      lat: Number(lat),
      lng: Number(lng),
      count: Math.round(Number(count || 0)),
      stopName,
      imageUrl,
    };

    setDensityMarkers((prev) => [...prev, entry]);
    setHeatmapPoints((prev) => [
      ...prev,
      { lat: entry.lat, lng: entry.lng, weight: entry.count },
    ]);

    if (map) drawDensityMarker(entry);
  }

  function addHeatPoint(lat, lng, weight) {
    setHeatmapPoints((prev) => [
      ...prev,
      { lat: Number(lat), lng: Number(lng), weight: Math.round(Number(weight || 0)) },
    ]);
  }

  // ======================================================
  // HEATMAP FETCH
  // ======================================================
  async function fetchHeatmapFromBackend() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/all-densities`);
      const data = await res.json();

      const formatted = data.map((s) => ({
        lat: s.lat,
        lng: s.lng,
        weight: Math.round(s.people || 0),
      }));

      setHeatmapPoints(formatted);
    } catch {
      // ignore fetch error when backend is offline
    }
  }

  useEffect(() => {
    fetchHeatmapFromBackend();
    const interval = setInterval(() => fetchHeatmapFromBackend(), 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!map) return;
    loadStopMarkers();
    densityMarkers.forEach((m) => drawDensityMarker(m));
  }, [map]);

  return (
    <MapContext.Provider
      value={{
        map,
        registerMap,
        addMarker,
        addHeatPoint,
        heatmapPoints,
        refreshHeatmap: fetchHeatmapFromBackend,
      }}
    >
      {children}
    </MapContext.Provider>
  );
}
