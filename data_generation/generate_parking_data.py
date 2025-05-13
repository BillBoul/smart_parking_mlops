import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

def get_parking_data(start_dt: datetime, end_dt: datetime, capacity: int = 8) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

    with open("config/semester_dates.json", "r") as f:
        semester_dates = json.load(f)
    with open("config/holidays.json", "r") as f:
        holidays = [datetime.strptime(d, "%Y-%m-%d").date() for d in json.load(f)]

    def get_semester_type(date):
        year = str(date.year)
        for semester, (start, end) in semester_dates.get(year, {}).items():
            if datetime.strptime(start, "%Y-%m-%d") <= date <= datetime.strptime(end, "%Y-%m-%d"):
                return semester
        return 'Break'

    def generate_event_days(year):
        return [datetime(year, random.randint(1, 12), random.randint(1, 28)).date() for _ in range(random.randint(5, 10))]

    event_days = {
        year: generate_event_days(year) for year in range(start_dt.year, end_dt.year + 1)
    }

    data = []
    current_date = start_dt.date()
    current_hour = start_dt.hour
    end_date = end_dt.date()
    end_hour = end_dt.hour

    while current_date <= end_date:
        for hour in range(7, 23):
            if current_date == start_dt.date() and hour < current_hour:
                continue
            if current_date == end_dt.date() and hour > end_hour:
                break

            day_of_week = current_date.weekday()
            is_weekend = int(day_of_week == 5)
            is_holiday = int(current_date in holidays)
            semester_type = get_semester_type(datetime.combine(current_date, datetime.min.time()))
            is_closed_day = int(is_holiday or day_of_week == 6 or semester_type == 'Break')
            event_nearby = int(current_date in event_days.get(current_date.year, []))

            if is_closed_day:
                available_spots = capacity
            elif is_weekend:
                available_spots = capacity - np.random.randint(0, capacity // 2)
            else:
                if 7 <= hour <= 9:
                    available_spots = max(0, capacity - np.random.randint(1, 3))
                elif 9 < hour <= 16:
                    available_spots = max(0, min(capacity, capacity + np.random.randint(-1, 2)))
                else:
                    available_spots = min(capacity, capacity + np.random.randint(1, 3))

            data.append({
                'date': current_date.strftime("%Y-%m-%d"),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'is_holiday': is_holiday,
                'is_closed_day': is_closed_day,
                'semester_type': semester_type,
                'event_nearby': event_nearby,
                'available_spots': available_spots
            })

        current_date += timedelta(days=1)

    return pd.DataFrame(data)
