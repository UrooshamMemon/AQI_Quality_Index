import hopsworks
import pandas as pd

from src.config import hopsworks_api_key, hopsworks_project


FEATURE_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "hour",
    "day",
    "month",
    "weekday",
    "aqi_difference",
    "aqi_change_rate",
    "aqi_moving_average",
    "rolling_temperature_average",
    "temperature_difference",
    "humidity_difference"
]


def load_latest_features():
    print("Connecting to Hopsworks...")

    project = hopsworks.login(
        project=hopsworks_project,
        api_key_value=hopsworks_api_key
    )

    feature_store = project.get_feature_store()

    feature_group = feature_store.get_feature_group(
        name="historical_aqi_features",
        version=1
    )

    df = feature_group.read(
        dataframe_type="pandas",
        read_options={
            "arrow_flight_config": {
                "timeout": 900
            }
        }
    )

    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time").reset_index(drop=True)

    latest_row = df.iloc[[-1]]

    return project, latest_row


def load_model(project, model_name):
    model_registry = project.get_model_registry()

    models = model_registry.get_models(
        name=model_name
    )

    if not models:
        raise RuntimeError(
            f"No registered model found: {model_name}"
        )

    latest_model = next(
        model for model in models
        if model.version == 1
    )

    print(
        f"Loading {model_name} "
        f"version {latest_model.version}..."
    )

    model_dir = latest_model.download()

    print(f"Model downloaded to: {model_dir}")

    import os
    import joblib

    model_file = os.path.join(
        model_dir,
        f"{model_name}.pkl"
    )

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Model file not found: {model_file}"
        )

    model = joblib.load(model_file)

    print(f"{model_name} loaded successfully.")

    return model


def make_prediction(model, features):
    X = features[FEATURE_COLUMNS]

    prediction = model.predict(X)

    return float(prediction[0])


def predict_all():

    project, latest_row = load_latest_features()

    model_24h = load_model(
        project,
        "aqi_forecast_24h"
    )

    model_48h = load_model(
        project,
        "aqi_forecast_48h"
    )

    model_72h = load_model(
        project,
        "aqi_forecast_72h"
    )

    forecast_24h = make_prediction(
        model_24h,
        latest_row
    )

    forecast_48h = make_prediction(
        model_48h,
        latest_row
    )

    forecast_72h = make_prediction(
        model_72h,
        latest_row
    )

    current_aqi = float(
        latest_row["aqi"].iloc[0]
    )

    return {
        "current": current_aqi,
        "24h": forecast_24h,
        "48h": forecast_48h,
        "72h": forecast_72h
    }


if __name__ == "__main__":

    print("\n======================================")
    print("AQI FORECAST PREDICTION")
    print("======================================")

    project, latest_row = load_latest_features()

    latest_time = latest_row["time"].iloc[0]

    print(f"\nLatest feature timestamp: {latest_time}")

    print("\nLoading registered models...")

    model_24h = load_model(
        project,
        "aqi_forecast_24h"
    )

    model_48h = load_model(
        project,
        "aqi_forecast_48h"
    )

    model_72h = load_model(
        project,
        "aqi_forecast_72h"
    )

    forecast_24h = make_prediction(
        model_24h,
        latest_row
    )

    forecast_48h = make_prediction(
        model_48h,
        latest_row
    )

    forecast_72h = make_prediction(
        model_72h,
        latest_row
    )

    current_aqi = float(
        latest_row["aqi"].iloc[0]
    )

    print("\n======================================")
    print("AQI FORECAST RESULTS")
    print("======================================")

    print(f"Current AQI:       {current_aqi:.2f}")
    print(f"24-hour forecast:  {forecast_24h:.2f}")
    print(f"48-hour forecast:  {forecast_48h:.2f}")
    print(f"72-hour forecast:  {forecast_72h:.2f}")

    print("\nPrediction completed successfully.")