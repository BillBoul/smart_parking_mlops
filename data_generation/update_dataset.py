import os
import pandas as pd
from datetime import datetime, timedelta
from generate_parking_data import get_parking_data
from generate_weather_data import get_weather_data, get_latest_weather_date

TARGET_CSV = "data/merged_parking_weather_dataset.csv"
DEFAULT_START = datetime(2021, 9, 1, 7)  # Start at 7:00 AM

def main():
    max_weather_date = get_latest_weather_date()
    if not os.path.exists(TARGET_CSV) or os.path.getsize(TARGET_CSV) == 0:
        print("📦 Initial dataset generation...")
        start_datetime = DEFAULT_START
    else:
        df_existing = pd.read_csv(TARGET_CSV)
        df_existing.dropna(how="all", inplace=True)

        if df_existing.empty:
            print("📦 Target dataset exists but is empty. Starting from scratch.")
            start_datetime = DEFAULT_START
        else:
            # Combine date + hour to detect last recorded hour
            df_existing['datetime'] = pd.to_datetime(df_existing['date'].astype(str) + ' ' + df_existing['hour'].astype(str) + ':00')
            latest_ts = df_existing['datetime'].max()
            start_datetime = latest_ts + timedelta(hours=1)
            print(f"🔄 Appending data from {start_datetime} to {max_weather_date}")

    if start_datetime > max_weather_date:
        print("✅ Dataset is already up-to-date.")
        return

    end_datetime = max_weather_date.replace(minute=0, second=0, microsecond=0)

    df_parking = get_parking_data(start_datetime, end_datetime)
    df_weather = get_weather_data(start_datetime, end_datetime)

    df_parking['date'] = pd.to_datetime(df_parking['date']).dt.strftime('%Y-%m-%d')
    df_weather['date'] = pd.to_datetime(df_weather['date']).dt.strftime('%Y-%m-%d')

    df_merged = pd.merge(df_parking, df_weather, on=['date', 'hour'], how='inner')

    if not os.path.exists(TARGET_CSV) or os.path.getsize(TARGET_CSV) == 0:
        df_merged.to_csv(TARGET_CSV, index=False)
        print(f"✅ Dataset initialized with {df_merged.shape[0]} rows.")
    else:
        df_existing.drop(columns=['datetime'], inplace=True, errors='ignore')
        df_final = pd.concat([df_existing, df_merged], ignore_index=True)
        df_final.to_csv(TARGET_CSV, index=False)
        print(f"Latest date in final dataset: {df_final['date'].max()} {df_final['hour'].max()}h")
        print(f"✅ Appended {df_merged.shape[0]} new rows.")

    print(df_merged.tail())

if __name__ == "__main__":
    main()
