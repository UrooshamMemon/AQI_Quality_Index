import requests
from src.config import latitude, longitude, historical_air_url
from datetime import date, timedelta

def historical_air_data(years=2):
    end_date = date.today()
    start_date = end_date - timedelta(365*years)

    params = {
        "latitude" : latitude,
        "longitude" : longitude,
        "start_date" : start_date.isoformat(),
        "end_date" : end_date.isoformat(),
        "hourly" : "us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    }

    response = requests.get(historical_air_url, params=params, timeout=10)

    try:
        response.raise_for_status()
        data = response.json()
        hourly_data = data['hourly']

        historical_data = {
            "time" : hourly_data.get('time'),
            "aqi" : hourly_data.get('us_aqi'),
            "pm10" : hourly_data.get('pm10'),
            "pm2_5" : hourly_data.get('pm2_5'),
            "carbon_monoxide" : hourly_data.get('carbon_monoxide'),
            "nitrogen_dioxide" : hourly_data.get('nitrogen_dioxide'),
            "sulphur_dioxide" : hourly_data.get('sulphur_dioxide'),
            "ozone" : hourly_data.get('ozone')
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