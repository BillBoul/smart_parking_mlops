import mlflow
import mlflow.keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from data_loader import load_dataset
from feature_engineering import add_time_features, prepare_features_and_labels, split_and_scale

# ==== CONFIG ====
CSV_PATH = "data/merged_parking_weather_dataset.csv"
EXPERIMENT_NAME = "Smart Parking Prediction"

# ==== 1. Load and preprocess test data ====
df = load_dataset(CSV_PATH)
df = add_time_features(df)
X, y = prepare_features_and_labels(df)
_, X_test_scaled, _, y_test = split_and_scale(X, y)

# ==== 2. Automatically find latest run from MLflow ====
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
latest_run = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
).iloc[0]

run_id = latest_run.run_id
model_uri = f"runs:/{run_id}/final_model"
print(f"Loading model from latest run ID: {run_id}")

# ==== 3. Load the model ====
model = mlflow.keras.load_model(model_uri)
print("Model loaded from MLflow.")

# ==== 4. Predict and round ====
y_pred = model.predict(X_test_scaled).flatten()
y_pred_rounded = np.clip(np.round(y_pred), 0, 8)

# ===
comparison_df = pd.DataFrame({
    "Actual": y_test.values[:20],
    "Predicted": y_pred_rounded[:20].astype(int)
})

print("\n Sample Predictions vs Actual:")
print(comparison_df.to_string(index=False))

# ==== Plot actual vs predicted over sample index ====
plt.figure(figsize=(12, 6))

plt.plot(range(100), y_test[:100], label='True Available Spots', marker='o')
plt.plot(range(100), y_pred_rounded[:100], label='Predicted Available Spots', marker='x')

plt.title('Validation Set: True vs Predicted Parking Availability (First 100 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Available Spots')
plt.legend()
plt.grid(True)
plt.show()
