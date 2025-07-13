# src/main.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==== Paths ====
DATA_PATH = Path(__file__).resolve().parent / "../data/train_FD001.txt"
OUT_DIR = Path(__file__).resolve().parent / "../output"
OUT_DIR.mkdir(exist_ok=True)

# ==== Load Data ====
column_names = ["engine_id", "cycle"] + \
    [f"op_setting_{i}" for i in range(1, 4)] + \
    [f"sensor_{i}" for i in range(1, 22)]

df = pd.read_csv(DATA_PATH, sep="\s+", header=None, names=column_names)

# ==== Calculate RUL ====
rul = df.groupby("engine_id")["cycle"].max().reset_index()
rul.columns = ["engine_id", "max_cycle"]
df = df.merge(rul, on="engine_id")
df["RUL"] = df["max_cycle"] - df["cycle"]
df.drop("max_cycle", axis=1, inplace=True)

# ==== Feature Selection ====
# Drop low-variance or constant sensors manually (basic)
drop_cols = ["op_setting_1", "op_setting_2", "op_setting_3",
             "sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
df = df.drop(drop_cols, axis=1)

# ==== Model Train ====
feature_cols = [c for c in df.columns if c not in ["engine_id", "cycle", "RUL"]]
X = df[feature_cols]
y = df["RUL"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ==== Residual Analysis ====
residuals = y_test - y_pred
mse = mean_squared_error(y_test, y_pred)
print(f"📊 MSE: {mse:.2f}")

plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Residuals of RUL Prediction")
plt.xlabel("Prediction Error (RUL)")
plt.tight_layout()
plt.savefig(OUT_DIR / "rul_residuals.png")

# ==== Save Predictions ====
pred_df = pd.DataFrame({
    "actual_RUL": y_test,
    "predicted_RUL": y_pred,
    "error": residuals
})
pred_df.to_csv(OUT_DIR / "rul_predictions.csv", index=False)

print("✅ Results saved in output/")
