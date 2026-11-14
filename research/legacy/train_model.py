import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Veriyi Yükle
df = pd.read_csv("counts.csv")

# 2. Feature ve Label oluştur
# Bir sonraki frame’deki kişi sayısını tahmin edeceğiz
df["next_count"] = df["count"].shift(-1)
df = df.dropna()

X = df[["count"]]          # Feature
y = df["next_count"]       # Hedef tahmin

# 3. Eğitim - Test verisini böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 4. Model oluştur
model = RandomForestRegressor(n_estimators=200, random_state=42)

# 5. Modeli eğit
model.fit(X_train, y_train)

# 6. Test setinde tahmin yap
preds = model.predict(X_test)

# 7. Hata oranını göster
mae = mean_absolute_error(y_test, preds)
print("Model MAE:", mae)

# 8. Modeli kaydet
joblib.dump(model, "model.pkl")

print("model.pkl oluşturuldu!")
