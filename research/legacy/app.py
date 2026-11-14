import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("İstasyon Yoğunluk Analizi - Bitirme Projesi")

# CSV dosyasını oku
df = pd.read_csv("counts.csv")

# Ham veri gösterimi
st.subheader("Ham Kişi Sayımı Grafiği")

st.line_chart(df['count'])

# İstatistikler
st.subheader("İstatistikler")

st.write(f"Toplam Frame: {len(df)}")
st.write(f"Ortalama Yoğunluk: {df['count'].mean():.2f}")
st.write(f"Maksimum Yoğunluk: {df['count'].max()}")
st.write(f"Minimum Yoğunluk: {df['count'].min()}")

# Yoğunluk kategorisi
st.subheader("Yoğunluk Durumu")

avg = df['count'].mean()

if avg < 2:
    st.success("🟢 Az Yoğun")
elif avg < 5:
    st.warning("🟡 Orta Yoğun")
else:
    st.error("🔴 Çok Yoğun")

# Grafikler
st.subheader("Frame-Kişi Sayısı Grafiği")

fig, ax = plt.subplots()
ax.plot(df['frame'], df['count'])
ax.set_xlabel("Frame")
ax.set_ylabel("Kişi Sayısı")

st.pyplot(fig)
import joblib

st.subheader("Makine Öğrenmesi Tahmini")

# modeli yükle
model = joblib.load("model.pkl")

# features (X)
X = df[["count"]]

# tahminler
df["predicted"] = model.predict(X)

# Gerçek vs Tahmin Grafiği
st.line_chart(df[["count", "predicted"]])

# Hata grafiği
df["error"] = abs(df["count"] - df["predicted"])

st.subheader("Tahmin Hatası Grafiği")
st.line_chart(df["error"])
