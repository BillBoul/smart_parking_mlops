from fastapi import FastAPI
import mlflow.pyfunc
import os
import sys
import pandas as pd
from api.utils import prepare_model_input_for_next_36_hours
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Add parent dir to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI(title="Smart Parking Prediction API")

# Load trained model
MODEL_PATH = "production_model"
model = mlflow.pyfunc.load_model(MODEL_PATH)

# Inject your actual WeatherAPI key
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

@app.get("/")
def root():
    return {"message": "Smart Parking Prediction API is running."}

@app.get("/predict")
def predict():
    try:
        # Generate input features and metadata
        X, df = prepare_model_input_for_next_36_hours(api_key=WEATHER_API_KEY)

        y_pred = model.predict(X).flatten()
        y_pred_rounded = np.clip(np.round(y_pred), 0, 8)
        df["predicted_availability"] = y_pred_rounded.astype(int)

        # Format and return response
        result = df[["date", "hour","is_closed_day","semester_type","temperature_C","rain_mm","wind_speed_kmh","humidity_percent", "predicted_availability"]].to_dict(orient="records")
        return {"predictions": result}

    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}
