import pandas as pd
from src.api.historical_weather_data import historical_weather_data
from src.api.historical_air_data import historical_air_data

def merge_historical_data():
    weather_history = historical_weather_data(2)
    air_history = historical_air_data(2)

    weather_df = pd.DataFrame(weather_history)
    air_df = pd.DataFrame(air_history)

    history_df = pd.merge(weather_df, air_df, on="time")

    return history_df