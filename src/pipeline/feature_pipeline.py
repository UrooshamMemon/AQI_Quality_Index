from src.api.fetch_weather import fetch_weather_data
from src.api.fetch_air_data import fetch_air_data

def create_features():
    weather_features = fetch_weather_data()
    air_features = fetch_air_data()
    
    features = weather_features | air_features

    return features