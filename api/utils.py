import pandas as pd
from datetime import datetime, timedelta
import json
import os
import requests
import joblib
from feature_engineering import add_time_features, prepare_features_and_labels


def generate_future_dataframe_for_tomorrow(api_key, location="Mansourieh,LB"):
    # 1. Define tomorrow's date and hourly slots
    tomorrow = datetime.today().date() + timedelta(days=1)
    hours = list(range(7, 23))  # 7 AM to 10 PM

    rows = []
    for hour in hours:
        rows.append({
            "date": tomorrow,
            "hour": hour
        })

    df = pd.DataFrame(rows)

    # 2. Day-based features
    df["day_of_week"] = df["date"].apply(lambda d: d.weekday())  # Monday=0
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 3. is_holiday from config
    holiday_path = os.path.join("config", "holidays.json")
    if os.path.exists(holiday_path):
        with open(holiday_path, "r") as f:
            holidays = json.load(f)
        holiday_dates = set(pd.to_datetime(holidays).date)
        df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
    else:
        df["is_holiday"] = 0

    # 4. semester_type from semester_dates.json
    semester_path = os.path.join("config", "semester_dates.json")
    df["semester_type"] = "Break"  # default to break if not in any semester

    if os.path.exists(semester_path):
        with open(semester_path, "r") as f:
            semesters = json.load(f)

        semester_periods = []
        for year in semesters:
            for sem_type, (start_str, end_str) in semesters[year].items():
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
                end = datetime.strptime(end_str, "%Y-%m-%d").date()
                semester_periods.append((start, end, sem_type.lower()))

        for i, row in df.iterrows():
            for start, end, sem_type in semester_periods:
                if start <= row["date"] <= end:
                    df.at[i, "semester_type"] = sem_type
                    break

    print("📅 Semester types in df:", df["semester_type"].unique())
    print("🧾 DataFrame columns at this point:", df.columns.tolist())


    # 5. is_closed_day
    df["is_closed_day"] = (
        (df["day_of_week"] == 6) |
        (df["is_holiday"] == 1) |
        (df["semester_type"].str.lower() == "break")
    ).astype(int)

    # 6. event_nearby
    df["event_nearby"] = 0

    # 7. Fetch and merge weather forecast
    weather_df = get_hourly_forecast_for_tomorrow(api_key, location)
    df = df.merge(weather_df, on="hour", how="left")

    # Reorder columns for consistency
    expected_columns = [
        "date", "hour", "day_of_week", "is_weekend", "is_holiday",
        "is_closed_day", "semester_type", "event_nearby",
        "temperature_C", "rain_mm", "wind_speed_kmh", "humidity_percent"
    ]
    df = df[expected_columns]

    return df


def get_hourly_forecast_for_tomorrow(api_key, location="Mansourieh,LB"):
    """
    Fetch hourly weather forecast for tomorrow from WeatherAPI.
    Returns a DataFrame for hours 7 to 22.
    """
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": api_key,
        "q": location,
        "days": 2,
        "aqi": "no",
        "alerts": "no"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"WeatherAPI error: {response.status_code} - {response.text}")

    data = response.json()
    try:
        forecast_hours = data["forecast"]["forecastday"][1]["hour"]
    except (KeyError, IndexError):
        raise Exception("Could not retrieve hourly forecast for tomorrow.")

    rows = []
    for hour_data in forecast_hours:
        hour = int(hour_data["time"].split(" ")[1].split(":")[0])
        if 7 <= hour <= 22:
            rows.append({
                "hour": hour,
                "temperature_C": hour_data["temp_c"],
                "humidity_percent": hour_data["humidity"],
                "wind_speed_kmh": hour_data["wind_kph"],
                "rain_mm": hour_data["precip_mm"]
            })

    return pd.DataFrame(rows)


def prepare_model_input_for_tomorrow(api_key, location="Mansourieh,LB"):
    """
    Combines feature generation, weather merging, and preprocessing.
    Returns:
        X_scaled: model-ready feature matrix
        df: original rows with metadata for output
    """
    df = generate_future_dataframe_for_tomorrow(api_key, location)

    df["available_spots"] = 0

    print("🧪 Before preprocessing:", df.columns.tolist())

    # Feature transformation (same as training)
    df_transformed = add_time_features(df)
    X, _ = prepare_features_and_labels(df_transformed)


    # Load and apply saved scaler
    scaler_path = os.path.join("production_model", "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Train the model first.")
    
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    return X_scaled, df
