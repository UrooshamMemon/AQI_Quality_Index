import pandas as pd

def engineering_features(df):
    df["time"] = pd.to_datetime(df["time"])

    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["weekday"] = df["time"].dt.dayofweek

    df["aqi_difference"] = df["aqi"].diff()

    df["aqi_change_rate"] = df["aqi"].pct_change()

    df["aqi_moving_average"] = df["aqi"].rolling(window=3).mean()
    df["rollinng_temperature_average"] = df["temperature_2m"].rolling(window=3).mean()

    df["temperature_difference"] = df["temperature_2m"].diff()
    df["humidity_difference"] = df["relative_humidity_2m"].diff()

    df = df.dropna()

    return df