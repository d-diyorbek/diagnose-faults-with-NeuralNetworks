import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Model va scaler yuklash
model = joblib.load("../modeling/model.pkl")
scaler = joblib.load("../modeling/scaler.pkl")

# Test faylni o‘qish
df = pd.read_csv("test_input_traffic_fixed.csv")

# Faqat signal ustunlari
X = df[[f"signal_{i+1}" for i in range(5)]]
y_true = df["out1"]

# Skalirlash
X_scaled = scaler.transform(X)

# Bashorat
y_pred = model.predict(X_scaled)[:, 0]  # faqat out1

# Aniqlik va tafsilotlar
acc = accuracy_score(y_true, y_pred)
print(f"🎯 Aniqlik (out1): {acc:.2%}")

for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        print(f"📌 Xatolik (qator {i}): haqiqiy = {y_true[i]}, model = {y_pred[i]}")

print("✅ Bashorat tugadi.")
