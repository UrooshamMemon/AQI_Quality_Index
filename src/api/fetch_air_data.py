import requests

from src.config import (
    latitude,
    longitude,
    historical_air_url
)

def fetch_air_data():

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "forecast_days": 1
    }

    response = requests.get(
        historical_air_url,
        params=params,
        timeout=10
    )

    try:
        response.raise_for_status()

        data = response.json()
        hourly_data = data["hourly"]

        air_data = {
            "aqi": hourly_data["us_aqi"][-1],
            "pm10": hourly_data["pm10"][-1],
            "pm2.5": hourly_data["pm2_5"][-1],
            "co": hourly_data["carbon_monoxide"][-1],
            "no2": hourly_data["nitrogen_dioxide"][-1],
            "o3": hourly_data["ozone"][-1],
            "so2": hourly_data["sulphur_dioxide"][-1],
            "nh3": None
        }

        return air_data

    except requests.exceptions.ConnectionError:
        print("Connection Error")

    except requests.exceptions.Timeout:
        print("Timeout")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")

    return None