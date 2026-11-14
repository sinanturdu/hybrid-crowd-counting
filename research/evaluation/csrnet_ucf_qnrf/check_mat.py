import scipy.io as sio
import os

# Örnek bir annotation dosyası
MAT_PATH = "Train/img_0001_ann.mat"

if not os.path.exists(MAT_PATH):
    raise FileNotFoundError(f"Dosya bulunamadı: {MAT_PATH}")

mat = sio.loadmat(MAT_PATH)

print("📌 MAT dosyası anahtarları:")
print(mat.keys())

print("\n📌 MAT içeriği (özet):")
for k in mat.keys():
    if not k.startswith("__"):
        print(f"\n🔹 {k}")
        print(type(mat[k]))
        try:
            print(mat[k].shape)
        except:
            pass
