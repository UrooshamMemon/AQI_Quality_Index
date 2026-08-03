import requests
from src.config import latitude, longitude, historical_weather_url
from datetime import date, timedelta

def historical_weather_data(years=2):
    end_date = date.today()
    start_date = end_date - timedelta(365*years)

    params = {
        "latitude" : latitude,
        "longitude" : longitude,
        "start_date" : start_date.isoformat(),
        "end_date" : end_date.isoformat(),
        "hourly" : "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
    }

    response = requests.get(historical_weather_url, params=params, timeout=10)

    try:
        response.raise_for_status()
        data = response.json()
        hourly_data = data['hourly']

        historical_data = {
             "time" : hourly_data.get('time'),
             "temperature_2m" : hourly_data.get('temperature_2m'),
             "relative_humidity_2m" : hourly_data.get('relative_humidity_2m'),
             "surface_pressure" : hourly_data.get('surface_pressure'),
             "wind_speed_10m" : hourly_data.get('wind_speed_10m')
        }

        return historical_data

    except requests.exceptions.ConnectionError:
            print("Connection Error")
    
    except requests.exceptions.Timeout:
        print("Time out")

    except requests.exceptions.HTTPError as e:
         print(f"HTTP Error: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")   