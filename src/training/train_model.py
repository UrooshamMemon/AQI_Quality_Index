import hopsworks
import pandas as pd
import numpy as np
import joblib
import os

from src.config import hopsworks_api_key, hopsworks_project

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ============================================================
# LOAD FEATURES FROM HOPSWORKS
# ============================================================

def load_features():

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

    df = df.sort_values("time").reset_index(drop=True)

    return df


# ============================================================
# CREATE FUTURE AQI TARGETS
# ============================================================

def create_targets(df):

    df["time"] = pd.to_datetime(df["time"])

    aqi_lookup = df[["time", "aqi"]].copy()

    # Future timestamps
    df["time_24h"] = df["time"] + pd.Timedelta(hours=24)
    df["time_48h"] = df["time"] + pd.Timedelta(hours=48)
    df["time_72h"] = df["time"] + pd.Timedelta(hours=72)

    # 24-hour target
    df = df.merge(
        aqi_lookup.rename(
            columns={
                "time": "time_24h",
                "aqi": "aqi_24h"
            }
        ),
        on="time_24h",
        how="left"
    )

    # 48-hour target
    df = df.merge(
        aqi_lookup.rename(
            columns={
                "time": "time_48h",
                "aqi": "aqi_48h"
            }
        ),
        on="time_48h",
        how="left"
    )

    # 72-hour target
    df = df.merge(
        aqi_lookup.rename(
            columns={
                "time": "time_72h",
                "aqi": "aqi_72h"
            }
        ),
        on="time_72h",
        how="left"
    )

    # Remove rows where future AQI is unavailable
    df = df.dropna(
        subset=[
            "aqi_24h",
            "aqi_48h",
            "aqi_72h"
        ]
    )

    # Remove temporary columns
    df = df.drop(
        columns=[
            "time_24h",
            "time_48h",
            "time_72h"
        ]
    )

    return df


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

def prepare_data(df):

    target_columns = [
        "aqi_24h",
        "aqi_48h",
        "aqi_72h"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in target_columns + ["time"]
    ]

    X = df[feature_columns]

    y = df[target_columns]

    print("\nFeatures used for training:")
    print(feature_columns)

    print("\nNumber of features:", len(feature_columns))

    print("\nTarget columns:")
    print(target_columns)

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining period:")
    print(
        X_train.index.min(),
        "to",
        X_train.index.max()
    )

    print("\nTesting period:")
    print(
        X_test.index.min(),
        "to",
        X_test.index.max()
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    target_column
):

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test[target_column],
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test[target_column],
            y_pred
        )
    )

    r2 = r2_score(
        y_test[target_column],
        y_pred
    )

    print(f"\n{target_column} Performance:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    return mae, rmse, r2


# ============================================================
# TRAIN 24-HOUR XGBOOST MODEL
# ============================================================

def train_24h_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n======================================")
    print("Training 24-hour XGBoost")
    print("======================================")

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )

    model.fit(
        X_train,
        y_train["aqi_24h"]
    )

    print("24-hour model training completed.")

    mae, rmse, r2 = evaluate_model(
        model,
        X_test,
        y_test,
        "aqi_24h"
    )

    return model, mae, rmse, r2


# ============================================================
# TRAIN 48-HOUR RIDGE MODEL
# ============================================================

def train_48h_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n======================================")
    print("Training 48-hour Ridge Regression")
    print("======================================")

    model = Ridge(
        alpha=1.0
    )

    model.fit(
        X_train,
        y_train["aqi_48h"]
    )

    print("48-hour model training completed.")

    mae, rmse, r2 = evaluate_model(
        model,
        X_test,
        y_test,
        "aqi_48h"
    )

    return model, mae, rmse, r2


# ============================================================
# TRAIN 72-HOUR RIDGE MODEL
# ============================================================

def train_72h_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n======================================")
    print("Training 72-hour Ridge Regression")
    print("======================================")

    model = Ridge(
        alpha=1.0
    )

    model.fit(
        X_train,
        y_train["aqi_72h"]
    )

    print("72-hour model training completed.")

    mae, rmse, r2 = evaluate_model(
        model,
        X_test,
        y_test,
        "aqi_72h"
    )

    return model, mae, rmse, r2


# ============================================================
# REGISTER MODEL IN HOPSWORKS
# ============================================================

def register_model(
    model,
    model_name,
    metrics
):

    print(f"\nRegistering {model_name}...")

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_file = f"models/{model_name}.pkl"

    joblib.dump(
        model,
        model_file
    )

    print(
        f"Model saved to: {model_file}"
    )

    project = hopsworks.login(
        project=hopsworks_project,
        api_key_value=hopsworks_api_key
    )

    mr = project.get_model_registry()

    hopsworks_model = mr.sklearn.create_model(
        name=model_name,
        metrics=metrics,
        description=(
            f"AQI forecasting model for "
            f"{model_name}"
        )
    )

    hopsworks_model.save(
        model_file
    )

    print(
        f"{model_name} registered successfully!"
    )

    print(
        f"Version: {hopsworks_model.version}"
    )

    return hopsworks_model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=============================================="
    )
    print(
        "AQI MODEL TRAINING AND REGISTRATION"
    )
    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading data from Hopsworks...")

    df = load_features()

    print(
        "Total rows:",
        len(df)
    )

    # --------------------------------------------------------
    # CREATE TARGETS
    # --------------------------------------------------------

    df = create_targets(df)

    print(
        "Rows after target creation:",
        len(df)
    )

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    X, y = prepare_data(df)

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # --------------------------------------------------------
    # TRAIN SELECTED MODELS
    # --------------------------------------------------------

    model_24h, mae_24h, rmse_24h, r2_24h = train_24h_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    model_48h, mae_48h, rmse_48h, r2_48h = train_48h_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    model_72h, mae_72h, rmse_72h, r2_72h = train_72h_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # FINAL MODEL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print(
        "=============================================="
    )
    print(
        "FINAL SELECTED MODELS"
    )
    print(
        "=============================================="
    )

    print(
        f"{'Target':<12}"
        f"{'Model':<25}"
        f"{'MAE':<10}"
        f"{'RMSE':<10}"
        f"{'R²':<10}"
    )

    print("-" * 67)

    print(
        f"{'aqi_24h':<12}"
        f"{'XGBoost':<25}"
        f"{mae_24h:<10.2f}"
        f"{rmse_24h:<10.2f}"
        f"{r2_24h:<10.4f}"
    )

    print(
        f"{'aqi_48h':<12}"
        f"{'Ridge Regression':<25}"
        f"{mae_48h:<10.2f}"
        f"{rmse_48h:<10.2f}"
        f"{r2_48h:<10.4f}"
    )

    print(
        f"{'aqi_72h':<12}"
        f"{'Ridge Regression':<25}"
        f"{mae_72h:<10.2f}"
        f"{rmse_72h:<10.2f}"
        f"{r2_72h:<10.4f}"
    )

    # --------------------------------------------------------
    # REGISTER MODELS
    # --------------------------------------------------------

    print("\n")
    print(
        "=============================================="
    )
    print(
        "REGISTERING MODELS IN HOPSWORKS"
    )
    print(
        "=============================================="
    )

    register_model(
        model_24h,
        "aqi_forecast_24h",
        {
            "mae": float(mae_24h),
            "rmse": float(rmse_24h),
            "r2": float(r2_24h)
        }
    )

    register_model(
        model_48h,
        "aqi_forecast_48h",
        {
            "mae": float(mae_48h),
            "rmse": float(rmse_48h),
            "r2": float(r2_48h)
        }
    )

    register_model(
        model_72h,
        "aqi_forecast_72h",
        {
            "mae": float(mae_72h),
            "rmse": float(rmse_72h),
            "r2": float(r2_72h)
        }
    )

    print("\n")
    print(
        "=============================================="
    )
    print(
        "ALL THREE MODELS REGISTERED SUCCESSFULLY!"
    )
    print(
        "=============================================="
    )