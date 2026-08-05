import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENWEATHER_API_KEY")
hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
hopsworks_project = os.getenv("HOPSWORKS_PROJECT")

weather_data_url = "https://api.openweathermap.org/data/2.5/weather"
air_data_url = "https://api.openweathermap.org/data/2.5/air_pollution"
historical_weather_url = "https://archive-api.open-meteo.com/v1/archive"
historical_air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

latitude = 24.8608
longitude = 67.0104