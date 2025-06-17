import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Datasetni o'qish
df = pd.read_csv(r"C:\Users\user\Desktop\Diagnose with Neural network\Diagnose faults V3\dataset\digital_fault_dataset_rule_based.csv")

# 2. X va Y ni ajratish
X = df[[f"signal_{i+1}" for i in range(5)]]
Y = df[[f"out{i+1}" for i in range(5)]]

# 3. Skalerni fit qilish
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Modelni yaratish va fit qilish
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
model = MultiOutputClassifier(base_model)
model.fit(X_scaled, Y)

# 5. Saqlash
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model va scaler saqlandi: modeling/model.pkl & scaler.pkl")
