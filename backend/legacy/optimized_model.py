from ultralytics import YOLO
import torch

class OptimizedModel:
    def __init__(self, model_path="models/best.pt"):
        print("📌 YOLO modeli yükleniyor...")

        # GPU varsa kullan
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔥 Kullanılan cihaz: {self.device}")

        # Modeli yükle
        self.model = YOLO(model_path)

        # GPU'ya taşı (varsa)
        try:
            self.model.to(self.device)
        except:
            pass

        # Hız için half precision (sadece GPU)
        self.use_half = False
        if self.device == "cuda":
            try:
                self.model.model.half()
                self.use_half = True
                print("⚡ Half precision aktif!")
            except:
                print("❗ Half precision desteklenmiyor.")

        # Warm-up (ilk istekte gecikmeyi yok eder)
        self.warmup()

    def warmup(self):
        print("🚀 Warm-up çalışıyor...")
        dummy = torch.zeros((1, 3, 640, 640))
        dummy = dummy.cuda() if self.device == "cuda" else dummy.cpu()

        try:
            _ = self.model(dummy)
        except:
            pass

        print("✅ Warm-up tamamlandı!")

    def predict(self, img_path):
        """Görsel üzerinde insan sayımı yapar"""
        results = self.model(img_path, device=self.device)
        return results
