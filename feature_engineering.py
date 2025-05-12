import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def add_time_features(df):
    """Adds cyclical time features and one-hot encodes semester_type."""
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df = pd.get_dummies(df, columns=['semester_type'])
    return df

def prepare_features_and_labels(df):
    """Selects relevant features and target for modeling."""
    features = [
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'is_weekend', 'is_holiday', 'is_closed_day', 'event_nearby',
        'temperature_C', 'rain_mm', 'wind_speed_kmh', 'humidity_percent'
    ]
    features += [col for col in df.columns if col.startswith('semester_type_')]

    X = df[features]
    y = df['available_spots']
    return X, y

def split_and_scale(X, y, test_size=0.2):
    """Splits the data and scales the features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
