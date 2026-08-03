import requests
from src.config import latitude, longitude, api_key, weather_data_url

def fetch_weather_data():
    params = {
        "lat" : latitude,
        "lon" : longitude,
        "appid" : api_key,
        "units": "metric"
    }

    response = requests.get(weather_data_url , params= params, timeout= 10)

    try:
        response.raise_for_status()   
        data = response.json()

        weather_data = {
            "weather" : data['weather'][0].get('main'),
            "temperature" : data['main'].get('temp'),
            "humidity" : data['main'].get('humidity'),
            "pressure" : data['main'].get('pressure'),
            "wind_speed" : data['wind'].get('speed')
        }
        return weather_data

    except requests.exceptions.ConnectionError:
        print("Connection Error")

    except requests.exceptions.Timeout:
        print("Time out")

    except requests.exceptions.HTTPError as e:
             print(f"HTTP Error: {e}")
    
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")  

