import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Make the project root importable (dashboard/ -> project_root/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_GROUP_NAME = "historical_aqi_features"
FEATURE_GROUP_VERSION = 1

# "display_name" is used by the Model Performance table so the model's
# human-friendly algorithm name lives in one place (here) instead of being
# duplicated in a separate lookup dict elsewhere in the file.
MODEL_REGISTRY = {
    "24h": {"name": "aqi_forecast_24h", "version": 1, "label": "24-Hour Forecast", "display_name": "XGBoost"},
    "48h": {"name": "aqi_forecast_48h", "version": 1, "label": "48-Hour Forecast", "display_name": "Ridge Regression"},
    "72h": {"name": "aqi_forecast_72h", "version": 1, "label": "72-Hour Forecast", "display_name": "Ridge Regression"},
}

POLLUTANT_COLUMNS = {
    "PM2.5": ["pm2_5", "pm25", "pm2.5"],
    "PM10": ["pm10"],
    "Ozone": ["ozone", "o3"],
    "CO": ["co", "carbon_monoxide"],
    "NO2": ["no2", "nitrogen_dioxide"],
    "SO2": ["so2", "sulphur_dioxide"],
}

WEATHER_COLUMNS = {
    "Temperature": ["temperature_2m"],
    "Humidity": ["relative_humidity_2m"],
    "Pressure": ["surface_pressure"],
    "Wind Speed": ["wind_speed_10m"],
}

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]

CACHE_TTL_SECONDS = 600  # 10 minutes — avoids hammering Hopsworks on every rerun

# ---------------------------------------------------------------------------
# SHAP explainability constants — must match the verified standalone script
# (dashboard/explainability.py) exactly: same feature group/version, same
# model/version, and the same exact training feature order.
# ---------------------------------------------------------------------------
SHAP_FEATURE_COLUMNS = [
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
    "humidity_difference",
]

SHAP_BACKGROUND_SAMPLE_SIZE = 300


# ---------------------------------------------------------------------------
# AQI helpers
# ---------------------------------------------------------------------------
def get_aqi_category(aqi_value):
    """Return (label, color) for a given AQI value using EPA-style breakpoints."""
    if aqi_value is None or pd.isna(aqi_value):
        return "Unknown", "#9e9e9e"
    aqi_value = float(aqi_value)
    for low, high, label, color in AQI_CATEGORIES:
        if low <= aqi_value <= high:
            return label, color
    if aqi_value > 500:
        return "Hazardous", "#7e0023"
    return "Unknown", "#9e9e9e"


def find_column(df, candidates):
    """Case-insensitive lookup of the first matching column name."""
    if df is None:
        return None
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# ADAPTER — EDIT THIS SECTION if your src/prediction/predict.py exposes
# differently-named functions. Everything else in this file calls the
# wrapper functions defined here, so this is the only place to change.
# ---------------------------------------------------------------------------
_predict_module = None
try:
    from src.prediction import predict as _predict_module  # noqa: E402
except Exception:
    _predict_module = None

try:
    from src import config as project_config  # noqa: E402
except Exception:
    project_config = None


def _hopsworks_login():
    import hopsworks
    from src.config import hopsworks_api_key, hopsworks_project

    return hopsworks.login(
        project=hopsworks_project,
        api_key_value=hopsworks_api_key
    )


def _try_reuse_predict_module():
    """
    Look for common function names in src/prediction/predict.py so we don't
    duplicate pipeline logic. Returns a dict of callables that were found
    (missing ones simply fall back to the direct-Hopsworks path below).
    """
    found = {}
    if _predict_module is None:
        return found

    feature_fn_names = ["load_latest_features", "get_latest_features", "fetch_features", "load_features"]
    model_fn_names = ["load_models", "get_models", "load_registered_models"]
    predict_fn_names = ["predict", "run_prediction", "make_predictions", "predict_all"]

    for name in feature_fn_names:
        if hasattr(_predict_module, name):
            found["features"] = getattr(_predict_module, name)
            break
    for name in model_fn_names:
        if hasattr(_predict_module, name):
            found["models"] = getattr(_predict_module, name)
            break
    for name in predict_fn_names:
        if hasattr(_predict_module, name):
            found["predict"] = getattr(_predict_module, name)
            break

    return found


_REUSED = _try_reuse_predict_module()


# ---------------------------------------------------------------------------
# Cached data / resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_fs_project():
    """Cached Hopsworks project handle (resource — not serialized)."""
    return _hopsworks_login()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Loading latest feature data from Hopsworks...")
def load_feature_data():
    """
    Load recent rows from the `historical_aqi_features` feature group.
    Tries to reuse a helper from predict.py first; falls back to a direct
    Hopsworks Feature Store read using the same feature group name/version.

    This is the SINGLE source of feature data for the whole app — the
    forecast cards, historical chart, pollutant panel, the EDA & Trends
    section, and the Prediction Explainability section all reuse this same
    cached DataFrame instead of issuing additional Hopsworks reads.
    """
    if "features" in _REUSED:
        df = _REUSED["features"]()
        if isinstance(df, pd.DataFrame):
            return df

    project = get_fs_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read(
    dataframe_type="pandas",
    read_options={
        "arrow_flight_config": {
            "timeout": 900
            }
        }
    )

    # Normalize a timestamp column if one exists, sort ascending
    ts_col = find_column(df, ["timestamp", "datetime", "date", "event_time", "time"])
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.sort_values(ts_col).reset_index(drop=True)

    return df


@st.cache_resource(show_spinner="Loading registered models from Hopsworks Model Registry...")
def load_models():
    """
    Load the three registered forecasting models. Tries to reuse a helper
    from predict.py first; falls back to the Hopsworks Model Registry API.

    In the fallback path, each bundle keeps the raw `meta` object returned
    by mr.get_model() (in addition to the downloaded `dir`), so downstream
    consumers — like the Model Performance section — can read registry
    metadata (e.g. stored training metrics) without another registry call.
    """
    if "models" in _REUSED:
        result = _REUSED["models"]()
        if result:
            return result

    project = get_fs_project()
    mr = project.get_model_registry()

    models = {}
    for horizon, info in MODEL_REGISTRY.items():
        model_meta = mr.get_model(name=info["name"], version=info["version"])
        model_dir = model_meta.download()
        models[horizon] = {"meta": model_meta, "dir": model_dir}
    return models


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Generating forecasts...")
def get_predictions():
    """
    Get all AQI predictions from the verified src/prediction/predict.py
    implementation.

    The dashboard does not perform its own fallback prediction, so the
    predictions shown here always use the same logic as predict.py.
    """
    if "predict" not in _REUSED:
        raise RuntimeError(
            "Could not import the prediction function from "
            "src/prediction/predict.py"
        )

    result = _REUSED["predict"]()

    if not isinstance(result, dict):
        raise RuntimeError(
            "predict.py did not return predictions as a dictionary."
        )

    return result


def refresh_all():
    """Clear all caches so the next render pulls fresh data from Hopsworks."""
    load_feature_data.clear()
    load_models.clear()
    get_predictions.clear()
    get_fs_project.clear()
    compute_shap_explanation.clear()
    get_model_performance_table.clear()


# ---------------------------------------------------------------------------
# SHAP explainability — data/compute layer
#
# This section intentionally does NOT touch src/prediction/predict.py, the
# training pipeline, or the feature pipeline. It reuses:
#   - the already-loaded feature DataFrame from load_feature_data()
#   - the already-cached model bundle from load_models() (same Hopsworks
#     Model Registry download used for the 24h forecast card)
# so no extra Hopsworks connections or Feature Store reads happen here.
# ---------------------------------------------------------------------------
def _load_24h_model_object():
    """
    Return the loaded aqi_forecast_24h model object, reusing the bundle
    already produced by load_models() (same logic/fallback path used by
    get_predictions()) instead of downloading or connecting again.
    """
    import joblib

    models = load_models()
    bundle = models.get("24h")
    if not bundle:
        raise RuntimeError("24h model bundle not found — load_models() did not return a '24h' entry.")

    model_dir = bundle["dir"] if isinstance(bundle, dict) else bundle

    model_file = None
    for f in os.listdir(model_dir):
        if f.endswith((".pkl", ".joblib")):
            model_file = os.path.join(model_dir, f)
            break
    if model_file is None:
        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.endswith((".pkl", ".joblib")):
                    model_file = os.path.join(root, f)
                    break
            if model_file:
                break
    if model_file is None:
        raise FileNotFoundError(f"No .pkl/.joblib model file found under {model_dir}")

    return joblib.load(model_file)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Computing SHAP explainability...")
def compute_shap_explanation(df):
    """
    Compute SHAP-based explainability for the aqi_forecast_24h model, reusing
    the already-loaded feature DataFrame (`df`, passed in from
    load_feature_data()) and the existing model-loading logic. No additional
    Hopsworks Feature Store reads or Hopsworks connections are made here.

    Mirrors the verified standalone script (dashboard/explainability.py):
    same feature order, same ~300-row background sample, same median fill
    for missing values, and shap.TreeExplainer.

    Returns a plain-data dict (safe for st.cache_data's hashing/serialization
    — we never try to cache SHAP Explanation objects or the model itself):
        {
            "global_importance": {feature: mean_abs_shap, ...},
            "individual_contributions": {feature: shap_value, ...},
            "predicted_24h_aqi": float,
            "current_aqi": float or None,
        }

    Raises on any failure so the caller can show a friendly warning instead
    of crashing the rest of the dashboard.
    """
    import shap  # local import: a missing/broken shap install only affects this section

    missing = [c for c in SHAP_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Feature group is missing expected columns for SHAP: {missing}")

    model = _load_24h_model_object()

    X = df[SHAP_FEATURE_COLUMNS].copy()
    X = X.fillna(X.median(numeric_only=True))

    sample_size = min(SHAP_BACKGROUND_SAMPLE_SIZE, len(X))
    X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    global_importance = dict(zip(SHAP_FEATURE_COLUMNS, mean_abs_shap.tolist()))

    latest_idx = df.index[-1]
    X_latest = X.loc[[latest_idx]]
    single_explanation = explainer(X_latest)
    individual_contributions = dict(zip(SHAP_FEATURE_COLUMNS, single_explanation.values[0].tolist()))

    predicted_24h_aqi = float(model.predict(X_latest)[0])

    aqi_col = find_column(df, ["aqi", "aqi_value", "current_aqi"])
    current_aqi = float(df.iloc[-1][aqi_col]) if aqi_col else None

    return {
        "global_importance": global_importance,
        "individual_contributions": individual_contributions,
        "predicted_24h_aqi": predicted_24h_aqi,
        "current_aqi": current_aqi,
    }


# ---------------------------------------------------------------------------
# Model Performance — data layer
#
# Reads the mae/rmse/r2 metrics that the training pipeline already stores on
# each registered model (via mr.sklearn.create_model(..., metrics=metrics))
# directly from the Hopsworks Model Registry. Nothing is hard-coded or
# recomputed here. Wherever load_models() already carries the registry
# metadata object (the same one used for forecasting/SHAP), that is reused
# instead of making another Model Registry call; a direct lookup only
# happens as a fallback, and it reuses the cached get_fs_project() Hopsworks
# connection rather than opening a new one.
# ---------------------------------------------------------------------------
def _get_model_registry_metadata(horizon, info):
    """
    Return the Hopsworks Model Registry metadata object for a given horizon.
    Prefers the `meta` object already present in the bundle returned by
    load_models(); falls back to a direct mr.get_model() lookup (still via
    the cached Hopsworks project) only if that isn't available.
    """
    models = load_models()
    bundle = models.get(horizon)

    meta = None
    if isinstance(bundle, dict):
        meta = bundle.get("meta")

    if meta is None:
        project = get_fs_project()
        mr = project.get_model_registry()
        meta = mr.get_model(name=info["name"], version=info["version"])

    return meta


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Loading model performance metrics...")
def get_model_performance_table():
    """
    Build the Model Performance table from the training-time metrics stored
    on each registered model in the Hopsworks Model Registry.

    Iterates MODEL_REGISTRY (the single source of model names/versions/
    display names) rather than duplicating that mapping elsewhere. Designed
    to keep working if MODEL_REGISTRY's versions are bumped in the future —
    it always reads whatever version is currently configured there.

    Returns a list of plain-data row dicts, one per horizon:
        {
            "horizon": "24h",
            "model_name": "aqi_forecast_24h",
            "display_name": "XGBoost",
            "version": 1,
            "mae": float or None,
            "rmse": float or None,
            "r2": float or None,
            "error": str or None,   # set (and other fields left None) if
                                     # metrics could not be retrieved
        }
    A failure for one model never raises — it's captured in that row's
    "error" field so the rest of the table (and the rest of the dashboard)
    still renders.
    """
    rows = []
    for horizon, info in MODEL_REGISTRY.items():
        row = {
            "horizon": horizon,
            "model_name": info["name"],
            "display_name": info.get("display_name", info["name"]),
            "version": info.get("version"),
            "mae": None,
            "rmse": None,
            "r2": None,
            "error": None,
        }
        try:
            meta = _get_model_registry_metadata(horizon, info)
            if meta is None:
                raise RuntimeError("No model metadata returned from the Model Registry.")

            metrics = getattr(meta, "training_metrics", None) or {}
            if not metrics:
                raise RuntimeError("Model metadata did not contain stored training metrics.")

            row["version"] = getattr(meta, "version", info.get("version"))
            row["mae"] = metrics.get("mae")
            row["rmse"] = metrics.get("rmse")
            row["r2"] = metrics.get("r2")

            if row["mae"] is None or row["rmse"] is None or row["r2"] is None:
                raise RuntimeError("One or more metrics (mae/rmse/r2) missing from stored metrics.")
        except Exception as e:
            row["error"] = str(e)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# UI — Sidebar
# ---------------------------------------------------------------------------
def render_epa_scale():
    # EPA AQI scale legend, shown in the sidebar.
    st.markdown("**EPA AQI Scale**")
    for low, high, label, color in AQI_CATEGORIES:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;margin-bottom:5px;">
                <div style="width:12px;height:12px;border-radius:50%;
                background-color:{color};margin-right:8px;flex-shrink:0;"></div>
                <div style="font-size:0.82rem;color:#000000;">
                    {low}–{high} &nbsp;·&nbsp; {label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar():
    with st.sidebar:
        st.markdown("**Dashboard - Uroosham Memon**")
        st.markdown("AQI Project - 10 pearls internship")
        st.markdown("---")

        render_epa_scale()

        st.markdown("---")

        st.markdown("**Last Updated**")
        st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        st.markdown("**Data Source**")
        st.write("Hopsworks Feature Store")
        st.caption(f"`{FEATURE_GROUP_NAME}` (v{FEATURE_GROUP_VERSION})")

        st.markdown("**Model Information**")
        for horizon, info in MODEL_REGISTRY.items():
            st.caption(f"`{info['name']}` — v{info['version']}")
        

        st.markdown("---")
        if st.button("Refresh Data", use_container_width=True):
            refresh_all()
            st.rerun()


# ---------------------------------------------------------------------------
# UI — Main sections (existing — unchanged behavior)
# ---------------------------------------------------------------------------
def render_header():
    st.title("AQI Forecast Dashboard - Karachi")
    st.caption("Air Quality Index forecasting — 24h / 48h / 72h ahead, powered by Hopsworks Feature Store & Model Registry")
    st.divider()


def render_metric_cards(predictions):
    current = predictions.get("current")
    h24 = predictions.get("24h")
    h48 = predictions.get("48h")
    h72 = predictions.get("72h")

    cols = st.columns(4)
    labels_values = [
        ("Current AQI", current),
        ("24-Hour Forecast", h24),
        ("48-Hour Forecast", h48),
        ("72-Hour Forecast", h72),
    ]

    for col, (label, value) in zip(cols, labels_values):
        category, color = get_aqi_category(value)
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color:#252525;
                    border:1px solid #2a2f38;
                    border-radius:12px;
                    padding:18px 16px;
                    text-align:center;
                ">
                    <div style="color:#9aa4b2;font-size:0.85rem;font-weight:600;letter-spacing:0.03em;text-transform:uppercase;">
                        {label}
                    </div>
                    <div style="font-size:2.2rem;font-weight:700;color:white;margin:6px 0 4px 0;">
                        {f"{value:.0f}" if value is not None else "—"}
                    </div>
                    <div style="
                        display:inline-block;
                        background-color:{color};
                        color:#111;
                        border-radius:999px;
                        padding:2px 12px;
                        font-size:0.78rem;
                        font-weight:700;
                    ">
                        {category}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# UI — AQI Health Alert (existing — unchanged behavior)
#
# Reuses the existing get_aqi_category() / AQI_CATEGORIES classification —
# no new/duplicate AQI thresholds are introduced here. Evaluates the current
# AQI plus the 24h/48h/72h forecasts already computed by get_predictions()
# and surfaces a warning for any horizon at "Unhealthy for Sensitive Groups"
# (AQI 101) or above. Purely presentational: makes no Hopsworks calls of its
# own and does not touch predictions, the feature DataFrame, or SHAP results.
# ---------------------------------------------------------------------------
def render_aqi_alerts(predictions):
    """
    Render the AQI Health Alert section.

    Evaluates Current AQI, 24h, 48h, and 72h forecasts (safely skipping any
    that are None/NaN) using the existing get_aqi_category() classification.
    Shows one alert card per horizon at AQI >= 101 ("Unhealthy for Sensitive
    Groups" or worse), or a single positive message if every available
    horizon is Good/Moderate (or if no predictions are available at all).
    """
    st.subheader("⚠️ AQI Health Alert")

    horizons = [
        ("Current AQI", predictions.get("current") if predictions else None),
        ("24-hour forecast", predictions.get("24h") if predictions else None),
        ("48-hour forecast", predictions.get("48h") if predictions else None),
        ("72-hour forecast", predictions.get("72h") if predictions else None),
    ]

    # Alert copy per category — layered on top of the existing
    # get_aqi_category() labels/colors, not a new classification system.
    ALERT_COPY = {
        "Unhealthy for Sensitive Groups": {
            "icon": "⚠️",
            "note": "Air quality may affect sensitive individuals.",
        },
        "Unhealthy": {
            "icon": "🚫",
            "note": "Everyone may begin to experience health effects; sensitive groups may experience more serious effects.",
        },
        "Very Unhealthy": {
            "icon": "🚨",
            "note": "Health alert: everyone may experience more serious health effects.",
        },
        "Hazardous": {
            "icon": "☠️",
            "note": "Health warning of emergency conditions — the entire population is more likely to be affected.",
        },
    }

    alerts = []
    for label, value in horizons:
        if value is None or pd.isna(value):
            continue
        category, color = get_aqi_category(value)
        if category in ALERT_COPY:
            alerts.append((label, float(value), category, color, ALERT_COPY[category]))

    if not alerts:
        st.markdown(
            """
            <div style="
                background-color:#111418;
                border:1px solid #2a2f38;
                border-left:4px solid #00e400;
                border-radius:10px;
                padding:14px 18px;
                color:#cfd6df;
                font-size:0.95rem;
            ">
                ✅ No unhealthy AQI levels detected in the current or forecasted values.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for label, value, category, color, copy in alerts:
        st.markdown(
            f"""
            <div style="
                background-color:#111418;
                border:1px solid #2a2f38;
                border-left:4px solid {color};
                border-radius:10px;
                padding:14px 18px;
                margin-bottom:10px;
            ">
                <div style="color:#f2f2f2;font-weight:700;font-size:1rem;margin-bottom:4px;">
                    {copy['icon']} AQI Health Alert
                </div>
                <div style="color:#cfd6df;font-size:0.92rem;line-height:1.5;">
                    {label}: <strong style="color:white;">{value:.0f}</strong><br/>
                    Status: <strong style="color:{color};">{category}</strong><br/>
                    {copy['note']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_forecast_chart(predictions):
    st.subheader("Forecast Trend")
    x_labels = ["Now", "+24h", "+48h", "+72h"]
    y_values = [
        predictions.get("current"),
        predictions.get("24h"),
        predictions.get("48h"),
        predictions.get("72h"),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=y_values,
            mode="lines+markers+text",
            text=[f"{v:.0f}" if v is not None else "" for v in y_values],
            textposition="top center",
            line=dict(color="#4da6ff", width=3),
            marker=dict(size=10, color="#4da6ff"),
            name="Forecasted AQI",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Time Horizon",
        yaxis_title="AQI",
        template="plotly_dark",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_historical_chart(df):
    st.subheader("Recent Historical AQI")
    ts_col = find_column(df, ["timestamp", "datetime", "date", "event_time", "time"])
    aqi_col = find_column(df, ["aqi", "aqi_value", "current_aqi"])

    if ts_col is None or aqi_col is None:
        st.info("Historical AQI chart unavailable — could not detect timestamp/AQI columns in the feature group.")
        return

    recent = df.tail(72)  # last ~72 readings
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent[ts_col],
            y=recent[aqi_col],
            mode="lines",
            line=dict(color="#ffb84d", width=2),
            fill="tozeroy",
            fillcolor="rgba(255,184,77,0.15)",
            name="Historical AQI",
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Time",
        yaxis_title="AQI",
        template="plotly_dark",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pollutants(df):
    st.subheader("Current Pollutant Levels")
    latest = df.iloc[-1]

    cols = st.columns(6)
    for col, (label, candidates) in zip(cols, POLLUTANT_COLUMNS.items()):
        col_name = find_column(df, candidates)
        value = latest[col_name] if col_name else None
        with col:
            st.metric(label, f"{value:.1f}" if pd.notna(value) else "—")


def render_timestamp(df):
    ts_col = find_column(df, ["timestamp", "datetime", "date", "event_time", "time"])
    if ts_col:
        latest_ts = pd.to_datetime(df.iloc[-1][ts_col])
        st.caption(f"Latest feature data timestamp: **{latest_ts.strftime('%Y-%m-%d %H:%M:%S')}**")
    else:
        st.caption("Latest feature data timestamp unavailable.")


# ---------------------------------------------------------------------------
# UI — EDA & Trends (existing — unchanged behavior)
#
# Everything here reuses the single `df` DataFrame already loaded by
# load_feature_data() in main() — no additional Hopsworks connections or
# Feature Store reads are made. Charts use Plotly with the same dark theme
# and color palette as the rest of the dashboard for visual consistency.
# ---------------------------------------------------------------------------
def render_eda_trend(df, ts_col, aqi_col):
    if ts_col is None:
        st.info("AQI trend chart unavailable — no timestamp column detected.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[ts_col],
            y=df[aqi_col],
            mode="lines",
            line=dict(color="#4da6ff", width=1.5),
            name="AQI",
        )
    )
    fig.update_layout(
        title="Full AQI History",
        xaxis_title="Time",
        yaxis_title="AQI",
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_eda_distribution(df, aqi_col):
    values = df[aqi_col].dropna()
    if values.empty:
        st.info("AQI distribution unavailable — no data.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=values, nbinsx=40, marker_color="#4da6ff"))
        fig.update_layout(
            title="AQI Distribution",
            xaxis_title="AQI",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=340,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Box(y=values, marker_color="#4da6ff", name="AQI"))
        fig.update_layout(
            title="AQI Boxplot",
            yaxis_title="AQI",
            template="plotly_dark",
            height=340,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    stats = values.describe()
    st.caption(
        f"Mean: **{stats['mean']:.1f}** · Median: **{values.median():.1f}** · "
        f"Std Dev: **{stats['std']:.1f}** · Min: **{stats['min']:.0f}** · Max: **{stats['max']:.0f}**"
    )


def render_eda_hourly_monthly(df, aqi_col):
    col1, col2 = st.columns(2)

    with col1:
        if "hour" in df.columns:
            hourly = df.groupby("hour")[aqi_col].mean().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=hourly["hour"], y=hourly[aqi_col], marker_color="#ffb84d"))
            fig.update_layout(
                title="Average AQI by Hour of Day",
                xaxis_title="Hour",
                yaxis_title="Mean AQI",
                template="plotly_dark",
                height=340,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Hourly pattern unavailable — 'hour' column not found.")

    with col2:
        if "month" in df.columns:
            monthly = df.groupby("month")[aqi_col].mean().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=monthly["month"], y=monthly[aqi_col], marker_color="#8f3f97"))
            fig.update_layout(
                title="Average AQI by Month",
                xaxis_title="Month",
                yaxis_title="Mean AQI",
                template="plotly_dark",
                height=340,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monthly pattern unavailable — 'month' column not found.")


def render_eda_pollutant_trends(df, ts_col):
    st.markdown("**Pollutant Trends Over Time**")
    x = df[ts_col] if ts_col else df.index
    colors = ["#4da6ff", "#ffb84d", "#8f3f97", "#00b894", "#ff7e00", "#e74c3c"]

    fig = go.Figure()
    plotted = False
    for (label, candidates), color in zip(POLLUTANT_COLUMNS.items(), colors):
        col_name = find_column(df, candidates)
        if col_name:
            fig.add_trace(go.Scatter(x=x, y=df[col_name], mode="lines", name=label, line=dict(color=color, width=1.3)))
            plotted = True

    if not plotted:
        st.info("Pollutant trend chart unavailable — no pollutant columns found.")
        return

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Concentration",
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_eda_scatter(df, aqi_col):
    st.markdown("**AQI vs Key Pollutants**")
    col1, col2 = st.columns(2)
    targets = [("PM2.5", POLLUTANT_COLUMNS["PM2.5"]), ("PM10", POLLUTANT_COLUMNS["PM10"])]

    for col, (label, candidates) in zip((col1, col2), targets):
        pol_col = find_column(df, candidates)
        with col:
            if pol_col is None:
                st.info(f"AQI vs {label} unavailable — column not found.")
                continue

            valid = df[[pol_col, aqi_col]].dropna()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=valid[pol_col],
                    y=valid[aqi_col],
                    mode="markers",
                    marker=dict(color="#00b894", size=5, opacity=0.5),
                    name="Observations",
                )
            )

            corr = float("nan")
            if len(valid) > 1:
                coeffs = np.polyfit(valid[pol_col], valid[aqi_col], 1)
                x_line = np.linspace(valid[pol_col].min(), valid[pol_col].max(), 50)
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=np.polyval(coeffs, x_line),
                        mode="lines",
                        line=dict(color="#ff4d4d", width=2),
                        name="Trend",
                    )
                )
                corr = valid.corr().iloc[0, 1]

            fig.update_layout(
                title=f"AQI vs {label} (corr = {corr:.2f})" if not pd.isna(corr) else f"AQI vs {label}",
                xaxis_title=label,
                yaxis_title="AQI",
                template="plotly_dark",
                height=360,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_eda_correlation(df, aqi_col):
    cols = [aqi_col]
    labels = ["AQI"]

    for label, candidates in POLLUTANT_COLUMNS.items():
        col_name = find_column(df, candidates)
        if col_name:
            cols.append(col_name)
            labels.append(label)

    for label, candidates in WEATHER_COLUMNS.items():
        col_name = find_column(df, candidates)
        if col_name:
            cols.append(col_name)
            labels.append(label)

    if len(cols) < 2:
        st.info("Correlation analysis unavailable — not enough numeric columns found.")
        return

    corr = df[cols].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="corr"),
        )
    )
    fig.update_layout(
        title="Correlation — AQI vs Pollutants & Weather",
        template="plotly_dark",
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_eda_section(df):
    st.subheader("EDA & Trends")
    st.caption(
        "Exploratory analysis of the historical AQI feature data already loaded from Hopsworks — "
        "no additional Feature Store calls are made for this section."
    )

    aqi_col = find_column(df, ["aqi", "aqi_value", "current_aqi"])
    ts_col = find_column(df, ["timestamp", "datetime", "date", "event_time", "time"])

    if aqi_col is None:
        st.info("EDA section unavailable — could not detect an AQI column in the feature data.")
        return

    tab_trend, tab_dist, tab_time, tab_pollutants, tab_corr = st.tabs(
        ["Trend", "Distribution", "Hourly / Monthly", "Pollutants", "Correlation"]
    )

    with tab_trend:
        render_eda_trend(df, ts_col, aqi_col)

    with tab_dist:
        render_eda_distribution(df, aqi_col)

    with tab_time:
        render_eda_hourly_monthly(df, aqi_col)

    with tab_pollutants:
        render_eda_pollutant_trends(df, ts_col)
        st.divider()
        render_eda_scatter(df, aqi_col)

    with tab_corr:
        render_eda_correlation(df, aqi_col)


# ---------------------------------------------------------------------------
# UI — Prediction Explainability (existing — unchanged behavior)
#
# Reuses the `df` DataFrame already loaded by load_feature_data() in main()
# and the model bundle already produced by load_models() — no additional
# Hopsworks Feature Store reads or Model Registry downloads happen here.
# SHAP computation is wrapped in compute_shap_explanation(), which is
# st.cache_data-cached (same TTL as the rest of the app) so it is not
# recomputed on every UI interaction, and any failure is caught here and
# shown as a friendly warning instead of crashing the dashboard.
# ---------------------------------------------------------------------------
def render_explainability_section(df):
    st.subheader("Prediction Explainability")
    st.write(
        "SHAP (SHapley Additive exPlanations) shows which features contribute most to the "
        "model's AQI prediction."
    )

    try:
        result = compute_shap_explanation(df)
    except Exception as e:
        st.warning(
            "⚠️ SHAP explainability isn't available right now, so this section can't be shown. "
            "The rest of the dashboard is unaffected."
        )
        with st.expander("Show technical details"):
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        return

    global_importance = result["global_importance"]
    individual_contributions = result["individual_contributions"]
    predicted_24h_aqi = result["predicted_24h_aqi"]
    current_aqi = result["current_aqi"]

    top_feature = max(individual_contributions, key=lambda k: abs(individual_contributions[k]))
    top_feature_shap = individual_contributions[top_feature]

    # --- Summary metrics -----------------------------------------------
    cols = st.columns(4)
    metric_pairs = [
        ("Current AQI", f"{current_aqi:.0f}" if current_aqi is not None else "—"),
        ("Predicted 24h AQI", f"{predicted_24h_aqi:.2f}"),
        ("Top Contributing Feature", top_feature),
        ("Its SHAP Contribution", f"{top_feature_shap:+.2f}"),
    ]
    for col, (label, value) in zip(cols, metric_pairs):
        with col:
            st.metric(label, value)

    st.write("")

    left, right = st.columns(2)

    # --- Global feature importance (top 10, mean |SHAP value|) ---------
    with left:
        global_series = pd.Series(global_importance).sort_values(ascending=False).head(10)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=global_series.values[::-1],
                y=global_series.index[::-1],
                orientation="h",
                marker_color="#4da6ff",
            )
        )
        fig.update_layout(
            title="Global Feature Importance (mean |SHAP value|)",
            xaxis_title="Mean |SHAP value|",
            template="plotly_dark",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Top 10 contributors to the latest 24h prediction (diverging) --
    with right:
        individual_series = pd.Series(individual_contributions)
        top10_idx = individual_series.abs().sort_values(ascending=False).head(10).index
        individual_series = individual_series.loc[top10_idx]
        colors = ["#ff4d4d" if v < 0 else "#00b894" for v in individual_series.values]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=individual_series.values[::-1],
                y=individual_series.index[::-1],
                orientation="h",
                marker_color=colors[::-1],
            )
        )
        fig.add_vline(x=0, line_color="#9aa4b2", line_width=1)
        fig.update_layout(
            title="Top Contributors to Latest 24h Prediction",
            xaxis_title="SHAP value (impact on prediction, AQI points)",
            template="plotly_dark",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Interpretation ---------------------------------------------------
    direction = "higher" if top_feature_shap > 0 else "lower"
    st.caption(
        f"**{top_feature}** is currently the strongest contributor to the 24-hour AQI prediction, "
        f"pushing it {direction} by about {abs(top_feature_shap):.2f} AQI points."
    )


# ---------------------------------------------------------------------------
# UI — Model Performance (new)
#
# Purely presentational: reads the plain-data rows produced by
# get_model_performance_table() (which itself reuses load_models()/
# get_fs_project() — no new Hopsworks connections here) and renders them as
# a table. Any model whose metrics couldn't be retrieved shows
# "Metrics unavailable" in that row instead of breaking the section.
# ---------------------------------------------------------------------------
def render_model_performance_section():
    st.subheader("Model Performance")

    rows = get_model_performance_table()

    table_rows = []
    for row in rows:
        if row["error"] is not None:
            table_rows.append(
                {
                    "Forecast": row["horizon"],
                    "Model": row["display_name"],
                    "Version": row["version"] if row["version"] is not None else "—",
                    "MAE": "Metrics unavailable",
                    "RMSE": "Metrics unavailable",
                    "R²": "Metrics unavailable",
                }
            )
        else:
            table_rows.append(
                {
                    "Forecast": row["horizon"],
                    "Model": row["display_name"],
                    "Version": row["version"],
                    "MAE": f"{row['mae']:.2f}",
                    "RMSE": f"{row['rmse']:.2f}",
                    "R²": f"{row['r2']:.4f}",
                }
            )

    perf_df = pd.DataFrame(table_rows)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.caption(
        "Performance metrics are calculated on the chronological test set used during model evaluation. "
        "Lower MAE/RMSE indicate lower prediction error, while higher R² indicates better explanatory performance."
    )

    if any(row["error"] is not None for row in rows):
        with st.expander("Show technical details for unavailable metrics"):
            for row in rows:
                if row["error"] is not None:
                    st.code(f"{row['horizon']} ({row['model_name']}): {row['error']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    render_sidebar()
    render_header()

    try:
        df = load_feature_data()
        predictions = get_predictions()
    except Exception as e:
        st.error(
            "❌ Could not connect to Hopsworks or load data/models. "
            "Check your Hopsworks API key / project configuration in `src/config.py` "
            "and your network connection."
        )
        with st.expander("Show technical details"):
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        st.stop()

    if df is None or df.empty:
        st.error(f"❌ Feature group `{FEATURE_GROUP_NAME}` (v{FEATURE_GROUP_VERSION}) returned no data.")
        st.stop()

    render_timestamp(df)
    st.write("")

    render_metric_cards(predictions)
    st.write("")

    render_aqi_alerts(predictions)  # reuses get_aqi_category()/AQI_CATEGORIES, no new Hopsworks calls
    st.write("")

    left, right = st.columns(2)
    with left:
        render_forecast_chart(predictions)
    with right:
        render_historical_chart(df)

    st.divider()
    render_pollutants(df)

    st.divider()
    render_eda_section(df)  # reuses the same `df` loaded above — no extra Hopsworks calls

    st.divider()
    render_explainability_section(df)  # reuses the same `df` and cached model bundle — no extra Hopsworks calls

    st.divider()
    render_model_performance_section()  # reuses the cached model bundle/registry metadata — no extra Hopsworks calls


if __name__ == "__main__":
    main()