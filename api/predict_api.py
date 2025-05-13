from fastapi import FastAPI
import mlflow.pyfunc
import os
import sys
import pandas as pd
from api.utils import prepare_model_input_for_tomorrow
import numpy as np

# Add parent dir to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI(title="Smart Parking Prediction API")

# Load trained model
MODEL_PATH = "production_model"
model = mlflow.pyfunc.load_model(MODEL_PATH)

# Inject your actual WeatherAPI key
WEATHER_API_KEY = "bf23b2217f8145e59b4220357251305"

@app.get("/")
def root():
    return {"message": "Smart Parking Prediction API is running."}

@app.get("/predict")
def predict():
    try:
        # Generate input features and metadata
        X, df = prepare_model_input_for_tomorrow(api_key=WEATHER_API_KEY)

        y_pred = model.predict(X).flatten()
        y_pred_rounded = np.clip(np.round(y_pred), 0, 8)
        df["predicted_availability"] = y_pred_rounded.astype(int)

        # Format and return response
        result = df[["date", "hour", "predicted_availability"]].to_dict(orient="records")
        return {"predictions": result}

    except Exception as e:
        return {"error": str(e)}
