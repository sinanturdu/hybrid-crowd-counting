import { useState } from "react";

export default function Photo() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [count, setCount] = useState(null);

  // Fotoğraf seçme
  const handleImage = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setCount(null);
  };

  // Analiz başlatma
  const analyzeImage = async () => {
    if (!image) return alert("Lütfen bir fotoğraf seçin!");

    setLoading(true);

    try {
      // Yakında burası backend YOLO API'ye bağlanacak
      await new Promise((res) => setTimeout(res, 1500));

      const fakeResult = Math.floor(20 + Math.random() * 50);
      setCount(fakeResult);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 animate-fadeIn">

      {/* BAŞLIK */}
      <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-200">
        Fotoğraf Analizi
      </h1>

      <p className="text-gray-500 dark:text-gray-300 mb-6">
        Yüklediğiniz fotoğraftaki kişi sayısını YOLO ile analiz edeceğiz.
      </p>

      {/* ORTA KUTU */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-6 flex flex-col md:flex-row gap-6">

        {/* SOL — Foto Yükleme */}
        <div className="w-full md:w-1/2 flex flex-col items-center">

          {/* PREVIEW */}
          {preview ? (
            <img
              src={preview}
              alt="preview"
              className="w-full rounded-lg shadow"
            />
          ) : (
            <div className="w-full h-64 bg-gray-100 dark:bg-gray-700 rounded-lg flex items-center justify-center text-gray-400">
              Fotoğraf seçilmedi
            </div>
          )}

          {/* Dosya Input */}
          <label className="mt-4 w-full text-center cursor-pointer bg-blue-600 text-white px-4 py-2 rounded-lg shadow hover:bg-blue-700">
            Fotoğraf Seç
            <input type="file" accept="image/*" className="hidden" onChange={handleImage} />
          </label>
        </div>

        {/* SAĞ — Sonuç */}
        <div className="w-full md:w-1/2 flex flex-col justify-between">

          {/* Analiz Sonucu */}
          <div className="text-center p-6 bg-gray-100 dark:bg-gray-700 rounded-lg shadow">
            {loading ? (
              <p className="text-lg text-blue-500 animate-pulse">
                Analiz ediliyor...
              </p>
            ) : count !== null ? (
              <p className="text-3xl font-bold text-green-600 dark:text-green-400">
                {count} Kişi Tespit Edildi
              </p>
            ) : (
              <p className="text-gray-500 dark:text-gray-300">Henüz analiz yapılmadı</p>
            )}
          </div>

          {/* ANALİZ BUTONU */}
          <button
            onClick={analyzeImage}
            disabled={!image || loading}
            className="mt-4 bg-purple-600 text-white px-6 py-3 rounded-lg shadow hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Analizi Başlat
          </button>
        </div>

      </div>
    </div>
  );
}
