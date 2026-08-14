import pandas as pd
import os
from src.pipeline.feature_pipeline import create_features
from src.storage.hopsworks_storage import get_previous_features
from src.new_features.feature_engineering import engineer_hourly_features
from src.config import enable_hopsworks

def run_hourly_pipeline():
    current_features = create_features()

    current_features["time"] = pd.Timestamp.now()

    current_df = pd.DataFrame([current_features])

    if enable_hopsworks:
        previous_df = get_previous_features()

        current_df = engineer_hourly_features(
            current_df,
            previous_df
        )

    else:
        print("Hopsworks disabled. Running on local server")

        file_path = "data/previous_features.csv"

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            previous_df = pd.read_csv(file_path)

            previous_df["time"] = pd.to_datetime(previous_df["time"])

            current_df = engineer_hourly_features(
                current_df,
                previous_df
            )

        else:
            print("No local previous data found.")
            print("Skipping historical-dependent features.")

    return current_df


def save_local_features(df):
    file_path = "data/previous_features.csv"

    os.makedirs("data", exist_ok=True)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:

        old_df = pd.read_csv(file_path)

        old_df["time"] = pd.to_datetime(old_df["time"])

        df = pd.concat([old_df, df], ignore_index=True)

    df.to_csv(file_path, index=False)

    print("Local features saved successfully.")