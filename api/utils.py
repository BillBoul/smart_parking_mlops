import pandas as pd
from datetime import datetime, timedelta
import json
import os
import requests
import joblib
from feature_engineering import add_time_features, prepare_features_and_labels


def generate_future_dataframe_for_next_36_hours(api_key, location="Mansourieh,LB"):
    now = datetime.now()
    current_hour = now.hour
    today = now.date()
    tomorrow = today + timedelta(days=1)

    # Include hours from next full hour today (if within range) + tomorrow
    hours_today = [h for h in range(current_hour + 1, 23) if 7 <= h <= 22]
    hours_tomorrow = [h for h in range(7, 23)]

    rows = []

    for hour in hours_today:
        rows.append({"date": today, "hour": hour})
    for hour in hours_tomorrow:
        rows.append({"date": tomorrow, "hour": hour})

    df = pd.DataFrame(rows)

    # Day-based features
    df["day_of_week"] = df["date"].apply(lambda d: d.weekday())
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Holiday feature
    holiday_path = os.path.join("config", "holidays.json")
    if os.path.exists(holiday_path):
        with open(holiday_path, "r") as f:
            holidays = json.load(f)
        holiday_dates = set(pd.to_datetime(holidays).date)
        df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
    else:
        df["is_holiday"] = 0

    # Semester type
    semester_path = os.path.join("config", "semester_dates.json")
    df["semester_type"] = "Break"

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

    df["is_closed_day"] = (
        (df["day_of_week"] == 6) |
        (df["is_holiday"] == 1) |
        (df["semester_type"].str.lower() == "break")
    ).astype(int)

    df["event_nearby"] = 0

    # Fetch and merge weather
    weather_df = get_hourly_forecast_for_next_36_hours(api_key, location)
    df = df.merge(weather_df, on=["date", "hour"], how="left")

    expected_columns = [
        "date", "hour", "day_of_week", "is_weekend", "is_holiday",
        "is_closed_day", "semester_type", "event_nearby",
        "temperature_C", "rain_mm", "wind_speed_kmh", "humidity_percent"
    ]
    df = df[expected_columns]

    return df


def get_hourly_forecast_for_next_36_hours(api_key, location="Mansourieh,LB"):
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": api_key,
        "q": location,
        "days": 2,  # today + tomorrow
        "aqi": "no",
        "alerts": "no"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"WeatherAPI error: {response.status_code} - {response.text}")

    data = response.json()
    rows = []

    for day_forecast in data["forecast"]["forecastday"]:
        for hour_data in day_forecast["hour"]:
            date = pd.to_datetime(hour_data["time"]).date()
            hour = pd.to_datetime(hour_data["time"]).hour
            if 7 <= hour <= 22:
                rows.append({
                    "date": date,
                    "hour": hour,
                    "temperature_C": hour_data["temp_c"],
                    "humidity_percent": hour_data["humidity"],
                    "wind_speed_kmh": hour_data["wind_kph"],
                    "rain_mm": hour_data["precip_mm"]
                })

    return pd.DataFrame(rows)


def prepare_model_input_for_next_36_hours(api_key, location="Mansourieh,LB"):
    df = generate_future_dataframe_for_next_36_hours(api_key, location)
    df["available_spots"] = 0  # placeholder

    df_transformed = add_time_features(df)
    X, _ = prepare_features_and_labels(df_transformed)

    scaler_path = os.path.join("production_model", "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Train the model first.")
    
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    return X_scaled, df
