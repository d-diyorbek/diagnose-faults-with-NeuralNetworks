import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# 1. Model va scaler yuklash
model = joblib.load("../modeling/model.pkl")
scaler = joblib.load("../modeling/scaler.pkl")

# 2. Test faylni yuklash
df = pd.read_csv("test_input_data_rule_based.csv")
X = df[[f"signal_{i+1}" for i in range(5)]]
Y_true = df[[f"out{i+1}" for i in range(5)]]

# 3. Skalirlash
X_scaled = scaler.transform(X)

# 4. Bashorat
Y_pred = model.predict(X_scaled)
Y_pred_df = pd.DataFrame(Y_pred, columns=[f"{col}_pred" for col in Y_true.columns])

# 5. Natijalarni birlashtirish va CSVga yozish
results = pd.concat([X, Y_true, Y_pred_df], axis=1)
results.to_csv("prediction_results.csv", index=False)

# 6. Har bir chiqish uchun aniqlik va xatolar
for col in Y_true.columns:
    acc = accuracy_score(Y_true[col], Y_pred_df[f"{col}_pred"])
    print(f"🎯 Aniqlik ({col}): {acc:.2%}")
    
    # Tafsilotli xatoliklar
    diff = Y_true[col].reset_index(drop=True) != Y_pred_df[f"{col}_pred"]
    if diff.any():
        print(f"📌 Xatoliklar ({col}):")
        for i in Y_true.index[diff]:
            real = Y_true.loc[i, col]
            pred = Y_pred_df.loc[i, f"{col}_pred"]
            print(f"  ↪ Qator {i}: haqiqiy = {real}, model = {pred}")
    else:
        print(f"✅ {col}: barcha bashoratlar to‘g‘ri!")

print("✅ Bashorat tugadi. Natijalar saqlandi: prediction_results.csv")
