# dataset/create_rule_based_dataset.py

import pandas as pd
import numpy as np

n_samples = 10000
X = np.random.rand(n_samples, 5)
df = pd.DataFrame(X, columns=[f"signal_{i+1}" for i in range(5)])

# Rule-based nosozliklar
df["out1"] = (df["signal_1"] > 0.8).astype(int)
df["out2"] = ((df["signal_2"] > 0.7) & (df["signal_3"] > 0.5)).astype(int)
df["out3"] = (df["signal_3"] < 0.2).astype(int)
df["out4"] = ((df["signal_4"] + df["signal_5"]) > 1.2).astype(int)
df["out5"] = ((df["signal_5"] > 0.3) & (df["signal_5"] < 0.4)).astype(int)

df.to_csv("digital_fault_dataset_rule_based.csv", index=False)
print("✅ Rule-based dataset created.")
