import requests
from src.config import latitude, longitude, api_key, air_data_url

def fetch_air_data():

    params = {
        "lat" : latitude,
        "lon" : longitude,
        "appid" : api_key
    }

    response = requests.get(air_data_url, params=params, timeout=10)

    try:
        response.raise_for_status()   
        data = response.json()
        
        main_data = data["list"][0].get('main')
        components = data["list"][0].get('components')

        air_data = {
            "aqi" : main_data.get('aqi'),
            "pm10" : components.get('pm10'),
            "pm2.5" : components.get('pm2_5'),
            "co" : components.get('co'),
            "no" : components.get('no'),
            "no2" : components.get('no2'),
            "o3" : components.get('o3'),
            "so2" : components.get('so2'),
            "nh3" : components.get('nh3')
        }
        return air_data
        
    except requests.exceptions.Timeout:
        print("Timeout")

    except requests.exceptions.ConnectionError:
        print("Connection Error")

    except requests.exceptions.HTTPError as e:
             print(f"HTTP Error: {e}")
    
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")  
