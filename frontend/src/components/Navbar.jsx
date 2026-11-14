// src/components/Navbar.jsx
import { useEffect, useState } from "react";
import { FiMoon, FiSun } from "react-icons/fi";

export default function Navbar() {
  const [dark, setDark] = useState(
    () => localStorage.getItem("darkMode") === "true"
  );

  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("darkMode", dark);
  }, [dark]);

  return (
    <header className="fixed top-0 left-64 right-0 h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between px-6 z-10">
      <div>
        <h2 className="text-lg font-semibold">Analiz Paneli</h2>
        <p className="text-xs text-gray-500">Kamera verilerinden yoğunluk analizi</p>
      </div>

      <button
        onClick={() => setDark((prev) => !prev)}
        className="w-9 h-9 rounded-full border border-gray-300 dark:border-gray-700 flex items-center justify-center hover:bg-gray-100 dark:hover:bg-gray-800 transition"
        aria-label="Tema değiştir"
      >
        {dark ? <FiSun /> : <FiMoon />}
      </button>
    </header>
  );
}
