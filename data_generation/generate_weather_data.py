from meteostat import Hourly
from datetime import datetime, timedelta
import pandas as pd

def get_weather_data(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    station_id = '40100'  # Beirut Airport
    df_weather = Hourly(station_id, start_dt, end_dt).fetch()

    df_weather.index.name = 'datetime'
    df_weather = df_weather.reset_index()

    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather['date'] = df_weather['datetime'].dt.strftime('%Y-%m-%d')
    df_weather['hour'] = df_weather['datetime'].dt.hour

    df_weather = df_weather[(df_weather['hour'] >= 7) & (df_weather['hour'] <= 22)]

    df_weather = df_weather[['date', 'hour', 'temp', 'prcp', 'wspd', 'rhum']]
    df_weather.rename(columns={
        'temp': 'temperature_C',
        'prcp': 'rain_mm',
        'wspd': 'wind_speed_kmh',
        'rhum': 'humidity_percent'
    }, inplace=True)

    df_weather['rain_mm'].fillna(0.0, inplace=True)

    return df_weather

def get_latest_weather_date(station_id='40100') -> datetime:
    from datetime import datetime, timedelta
    from meteostat import Hourly

    now = datetime.now()

    for delta in range(0, 30):
        date = now - timedelta(days=delta)
        df = Hourly(station_id, date, date + timedelta(days=1)).fetch()
        if not df.empty:
            latest = df.index[-1].to_pydatetime()
            if latest <= now:
                return latest.replace(minute=0, second=0, microsecond=0)
    raise Exception("Unable to determine latest weather data available.")

