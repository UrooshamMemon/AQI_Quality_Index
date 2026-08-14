import pandas as pd

def engineer_historical_features(df):
    df["time"] = pd.to_datetime(df["time"])

    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["weekday"] = df["time"].dt.dayofweek

    df["aqi_difference"] = df["aqi"].diff()

    df["aqi_change_rate"] = df["aqi"].pct_change()

    df["aqi_moving_average"] = df["aqi"].rolling(window=3).mean()
    df["rolling_temperature_average"] = df["temperature_2m"].rolling(window=3).mean()

    df["temperature_difference"] = df["temperature_2m"].diff()
    df["humidity_difference"] = df["relative_humidity_2m"].diff()

    df = df.dropna()

    return df


def engineer_hourly_features(current_features, previous_df):
    current = current_features.copy()

    current["time"] = pd.to_datetime(current["time"])

    current["hour"] = current["time"].dt.hour
    current["day"] = current["time"].dt.day
    current["month"] = current["time"].dt.month
    current["weekday"] = current["time"].dt.dayofweek

    previous_aqi = previous_df["aqi"].iloc[-1]

    current["aqi_difference"] = current["aqi"] - previous_aqi

    current["aqi_change_rate"] = (
        (current["aqi"] - previous_aqi) / previous_aqi
        if previous_aqi != 0 else 0
    )

    aqi_values = list(previous_df["aqi"].tail(2)) + [current["aqi"]]
    current["aqi_moving_average"] = sum(aqi_values) / len(aqi_values)

    temperature_values = (
        list(previous_df["temperature_2m"].tail(2))
        + [current["temperature_2m"]]
    )
    current["rolling_temperature_average"] = (
        sum(temperature_values) / len(temperature_values)
    )

    current["temperature_difference"] = (
        current["temperature_2m"]
        - previous_df["temperature_2m"].iloc[-1]
    )

    current["humidity_difference"] = (
        current["relative_humidity_2m"]
        - previous_df["relative_humidity_2m"].iloc[-1]
    )

    return current