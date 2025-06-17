import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import io

st.set_page_config(page_title="Digital Fault Diagnoser", layout="wide")

st.title("🧠 Digital System Fault Diagnosis")
st.write("Upload your test signal data below to detect faults using the trained AI model.")

# Model va scaler yuklash
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Fayl yuklash
uploaded_file = st.file_uploader("📂 Upload a CSV file with signals", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Tekshirish uchun kerakli ustunlar borligini tekshiramiz
    expected_signal_cols = [f"signal_{i+1}" for i in range(5)]
    expected_output_cols = [f"out{i+1}" for i in range(5)]

    if not all(col in df.columns for col in expected_signal_cols + expected_output_cols):
        st.error("❌ CSV fayl format noto‘g‘ri. Quyidagi ustunlar bo‘lishi kerak: " + ", ".join(expected_signal_cols + expected_output_cols))
    else:
        # Bashorat qilish
        X = df[expected_signal_cols]
        Y_true = df[expected_output_cols]
        X_scaled = scaler.transform(X)
        Y_pred = model.predict(X_scaled)
        Y_pred_df = pd.DataFrame(Y_pred, columns=[f"{col}_pred" for col in Y_true.columns])

        # Natijalarni birlashtirish
        results = pd.concat([X, Y_true, Y_pred_df], axis=1)

        # Har bir chiqish bo‘yicha aniqlik va xatoliklarni chiqarish
        st.subheader("📊 Prediction Results")

        for col in Y_true.columns:
            acc = accuracy_score(Y_true[col], Y_pred_df[f"{col}_pred"])
            st.markdown(f"**🎯 {col.upper()} Aniqlik:** `{acc:.2%}`")

            diff = Y_true[col].reset_index(drop=True) != Y_pred_df[f"{col}_pred"]
            if diff.any():
                st.markdown(f"**📌 Xatoliklar ({col}):**")
                for i in Y_true.index[diff][:10]:  # Faqat dastlabki 10 ta xatolikni ko‘rsatamiz
                    real = Y_true.loc[i, col]
                    pred = Y_pred_df.loc[i, f"{col}_pred"]
                    st.write(f"🔻 Qator {i}: Haqiqiy = {real}, Model = {pred}")
            else:
                st.success(f"✅ {col}: barcha bashoratlar to‘g‘ri!")

        # CSV saqlash opsiyasi
        csv_data = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Natijalarni yuklab olish (CSV)", data=csv_data, file_name="prediction_results.csv", mime="text/csv")
