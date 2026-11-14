# Hibrit Kalabalık Sayımı ve Toplu Taşıma Yoğunluk Analizi (YOLOv8 + CSRNet)

İstanbul toplu taşıma duraklarında kalabalık sayımı, yoğunluk tahmini ve harita tabanlı görselleştirme için geliştirilmiş bitirme projesi. Sistem; düşük/orta yoğunluklu sahneler için nesne tespiti (**YOLOv8**) ve yüksek yoğunluklu kalabalıklar için genişletilmiş evrişimli yoğunluk haritası regresyonu (**CSRNet**) modellerini birleştirir. **FastAPI** tabanlı arka uç (backend) ve **React + Vite + Tailwind CSS + Chart.js + Google Maps / deck.gl** tabanlı kullanıcı arayüzünden oluşur.

> **Dil / Language:** [Türkçe (docs/README.tr.md)](README.tr.md) | [English (../README.md)](../README.md)

---

## Proje Klasör Yapısı

```text
hybrid-crowd-counting/
├── backend/                        # FastAPI uygulama sunucusu
│   ├── main.py                     # Ana API (YOLOv8 + CSRNet çıkarımı ve durak yoğunluk uç noktaları)
│   ├── models/
│   │   ├── csrnet.py               # PyTorch CSRNet mimarisi (VGG16 + dilated conv)
│   │   └── __init__.py             # (best.pt, best2.pt, csrnet_AB_best.pth buraya konur; Git dışı)
│   ├── tests/
│   │   └── test_security_and_api.py# Güvenlik (path traversal, CORS) ve model entegrasyon testleri
│   ├── legacy/                     # Eski Node.js/Express prototipi (arşiv amaçlı)
│   ├── uploads/                    # Geçici yükleme klasörü (Git dışı)
│   ├── processed/                  # İşlenmiş görsel ve video çıktı klasörü (Git dışı)
│   ├── requirements.txt            # Doğrulanmış Python bağımlılıkları
│   └── requirements-lock.txt       # Python 3.12.7 + CUDA 12.1 tam sürüm kilidi
├── frontend/                       # React 19 + Vite 7 + Tailwind CSS + Chart.js + Google Maps arayüzü
│   ├── src/
│   │   ├── config.js               # Ortam değişkeni tabanlı API URL ve Google Maps ayarları
│   │   ├── context/MapContext.jsx  # Durak işaretçileri (marker) ve ısı haritası (heatmap) yönetimi
│   │   ├── data/stops.js           # 15 İstanbul toplu taşıma durağının koordinatları
│   │   ├── components/             # Navbar (karanlık mod destekli) ve Sidebar
│   │   └── pages/                  # Dashboard, ImageAnalysis, VideoAnalysis, MapPage
│   └── public/stops/               # Durak bilgi penceresi görselleri (Git dışı)
├── research/                       # Makine öğrenmesi eğitim, değerlendirme ve veri ön işleme kodları
│   ├── training/                   # CSRNet (ShanghaiTech A+B, UCF-QNRF) ve YOLO eğitim betikleri
│   ├── evaluation/                 # Değerlendirme betikleri, eğitim grafikleri ve sonuç tabloları
│   ├── preprocessing/              # Veri seti dönüştürme betikleri (CrowdHuman, MOT20, ShanghaiTech)
│   ├── configs/                    # Veri seti YAML dosyaları ve kaynak bilgileri
│   └── legacy/                     # İlk aşama Streamlit / RandomForest denemeleri
├── examples/                       # Telif hakkı doğrulanmış örnek görsel/videolar için klasör
├── docs/                           # Teknik dokümantasyon
│   ├── README.tr.md                # Türkçe ana doküman
│   ├── ARCHITECTURE.md             # Sistem mimarisi ve API sözleşmeleri
│   ├── MODELS_AND_EXPERIMENTS.md   # Eğitilen modeller, metrikler ve büyük dosya envanteri
│   └── SECURITY_AND_LICENSING.md   # Güvenlik iyileştirmeleri, API anahtarı ve lisans incelemesi
├── .env.example                    # Ortam değişkeni şablonu
└── .gitignore                      # Model ağırlıkları, veri setleri ve gizli dosyalar için kurallar
```

---

## Doğrulanmış Özellikler

1. **Fotoğraf Analizi (`/image` -> `POST /upload-image`):**
   - Seçilen durağa ait fotoğrafı üç farklı model seçeneğiyle analiz eder:
     - `best`: `reid.v1i.yolov8` veri seti üzerinde eğitilmiş YOLOv8s modeli.
     - `best2`: Yerel insan tespiti veri seti üzerinde 80 epoch eğitilmiş YOLOv8s modeli.
     - `csrnet_ab`: ShanghaiTech Part A + Part B birleşik veri seti üzerinde eğitilmiş PyTorch CSRNet modeli ($384 \times 384$ yama tabanlı çıkarım, $\text{stride} = 192$, Gauss ağırlıklı birleştirme ve JET ısı haritası katmanı).
   - Tespit edilen kişi sayısını, işlenmiş görseli ve harita üzerindeki durak yoğunluk işaretçisini günceller.
2. **Video Analizi ve Grafik Senkronizasyonu (`/video` -> `POST /video`):**
   - Yüklenen videoları YOLOv8 ile kare kare (veya CSRNet ile her 5 karede bir) işler ve FFmpeg (`libx264`, `yuv420p`, `+faststart`) ile tarayıcı uyumlu MP4 formatına dönüştürür.
   - Video oynatılırken o anki karedeki kişi sayısını **Chart.js** çizgi grafiği üzerinde eşzamanlı olarak vurgular.
3. **Durak Haritası ve Isı Haritası (`/map`):**
   - İstanbul'daki 15 merkezi toplu taşıma noktasını Google Maps Gelişmiş İşaretçiler (*Advanced Markers*) ve açılıp kapatılabilir **deck.gl `HeatmapLayer`** ile görselleştirir.
   - Her 10 saniyede bir `/api/all-densities` uç noktasından güncel yoğunlukları çeker.

---

## Kurulum ve Çalıştırma

### 1. Model Ağırlıklarının Yerleştirilmesi

Model ağırlıkları (`~105 MB`) boyut ve veri seti lisans koşulları nedeniyle Git deposuna dahil edilmemiştir. Aşağıdaki üç dosyayı `backend/models/` dizinine yerleştirin:

- `backend/models/best.pt` (`22.5 MB`, YOLOv8s — `train14`)
- `backend/models/best2.pt` (`22.5 MB`, YOLOv8s — `yolo8/train`)
- `backend/models/csrnet_AB_best.pth` (`65.1 MB`, PyTorch CSRNet — `ShanghaiTech A+B`)

> **Temiz Klon (Clean Clone) Davranışı:**
> - Model ağırlıkları henüz `backend/models/` klasörüne yerleştirilmeden sunucu başlatılırsa, FastAPI uygulaması sorunsuz ayağa kalkar; analiz uç noktaları (`POST /upload-image`, `POST /video`) çağrıldığında ağırlık dosyasının eksik olduğunu belirten anlaşılır bir `503 Service Unavailable` yanıtı döner.
> - `frontend/public/stops/*.jpg` altındaki durak fotoğrafları telif/yeniden dağıtım izinleri doğrulanmadığı için Git deposuna dahil edilmemiştir. Temiz bir klonda harita işaretçisine tıklandığında eksik görsel alanı otomatik olarak gizlenir ve durak adı, yoğunluk etiketi ile kişi sayısı sorunsuz görüntülenir.

### 2. Backend (FastAPI) Kurulumu

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate

# NVIDIA GPU (CUDA 12.1) kullanıyorsanız önce PyTorch kurun:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Diğer bağımlılıkları yükleyin:
pip install -r requirements.txt

# Sunucuyu başlatın:
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 3. Frontend (React + Vite) Kurulumu

```powershell
cd frontend
cp .env.example .env
# .env dosyasındaki VITE_GOOGLE_MAPS_API_KEY değerini kendi kısıtlanmış anahtarınızla güncelleyin

npm ci
npm run dev
```

Tarayıcıda `http://localhost:5173` adresini açın.

---

## Güvenlik Notları

- **Frontend Ortam Değişkenleri Gizli Değildir:** Vite projelerinde `VITE_` önekiyle tanımlanan değişkenler derleme sırasında doğrudan istemci tarafı JavaScript paketine gömülür. `.env` kullanımı anahtarın Git geçmişine girmesini engeller ancak tarayıcıda görünmesini engellemez. Bu nedenle Google Cloud Console üzerinden API anahtarınıza **HTTP yönlendiren (referrer) kısıtlaması** ve **yalnızca Maps JavaScript API yetkisi** tanımlamanız zorunludur.
- **Güvenli Dosya Sunumu:** `/file/{path}` uç noktası yalnızca `backend/processed/` altındaki `.jpg`, `.jpeg`, `.png` ve `.mp4` dosyalarını sunar; dizin geçişi (`..`) ve kaynak kod erişim girişimlerini HTTP `403` ile engeller.
- **CORS Kısıtlaması:** Tüm kaynaklara açık (`*`) CORS ayarı kaldırılmış, `CORS_ALLOWED_ORIGINS` ile yalnızca izin verilen adreslere sınırlandırılmıştır.
