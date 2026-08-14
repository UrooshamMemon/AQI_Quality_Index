from src.api.fetch_weather import fetch_weather_data
from src.api.fetch_air_data import fetch_air_data

def create_features():
    weather_features = fetch_weather_data()
    air_features = fetch_air_data()
    
    features = weather_features | air_features

    features["temperature_2m"] = features.pop("temperature")
    features["relative_humidity_2m"] = features.pop("humidity")
    features["surface_pressure"] = features.pop("pressure")
    features["wind_speed_10m"] = features.pop("wind_speed")

    return features