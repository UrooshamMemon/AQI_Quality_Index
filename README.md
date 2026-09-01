# AQI Forecasting System

An end-to-end **Air Quality Index (AQI) forecasting system** developed during my **10 Pearls Data Science Internship**. The system collects real-time weather and air-quality data, performs feature engineering, stores features in Hopsworks, trains machine-learning models for multiple forecast horizons, and provides an interactive Streamlit dashboard for monitoring current and forecasted AQI.

The system predicts AQI for the next **24, 48, and 72 hours**.

## Project Report
[View Full Project Report (PDF)](AQI_Forecasting_Report.pdf)

---

## Project Overview

Air pollution can change significantly over time due to weather conditions and pollutant concentrations. This project uses historical and real-time environmental data to build a machine-learning pipeline capable of forecasting future AQI levels.

The complete system follows this workflow:

Weather & Air Quality APIs
          ↓
   Data Collection
          ↓
   Feature Engineering
          ↓
 Hopsworks Feature Store
          ↓
   Model Training
          ↓
   Model Evaluation
          ↓
 Hopsworks Model Registry
          ↓
   Prediction Pipeline
          ↓
 Streamlit Dashboard

The system is automated so that new environmental data is continuously collected and forecasting models can be retrained using updated data.

---

## Key Features

### Feature Pipeline Development

The system collects environmental data from external APIs (OpenWeather & OpenMeteo) and transforms raw data into machine-learning features.

Data includes:

* Temperature
* Relative humidity
* Atmospheric pressure
* Wind speed
* AQI
* PM2.5
* PM10
* Carbon monoxide
* Nitrogen dioxide
* Sulphur dioxide
* Ozone

Time-based features include:

* Hour
* Day
* Month
* Weekday

Derived features include:

* AQI difference
* AQI change rate
* AQI moving average
* Rolling temperature average
* Temperature difference
* Humidity difference

A total of **21 features** are used for model training.

---

### Hopsworks Feature Store

Processed environmental features are stored in the **Hopsworks Feature Store**.

The Feature Store provides centralized storage for:

* Historical training data
* Engineered features
* Feature retrieval
* Consistent data access between training and prediction

Feature group:
historical_aqi_features


---

## Machine Learning Models

Separate forecasting models are trained for each forecasting horizon.

### Selected Model Performance

The final selected models were evaluated using MAE, RMSE, and R².

| Forecast Horizon | Model            |      MAE |      RMSE |         R² |
| ---------------- | ---------------- | -------: | --------: | ---------: |
| 24 hours         | XGBoost          |   6.17   |    9.21   |   0.5174   |
| 48 hours         | Ridge Regression |   8.66   |   12.05   |   0.1711   |
| 72 hours         | Ridge Regression |   9.18   |   12.88   |   0.0577   |

The models are evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

The trained models are stored in the **Hopsworks Model Registry**.

Registered models:
aqi_forecast_24h
aqi_forecast_48h
aqi_forecast_72h


---

### Persistence Baseline

A **persistence baseline** is used as a benchmark for evaluating the forecasting models.

The persistence approach assumes that the future AQI will remain equal to the current AQI:
Predicted future AQI = Current AQI

This provides a simple reference point for determining whether the machine-learning models provide an improvement over a basic forecasting strategy.

### Persistence Baseline Performance

| Forecast Horizon |   MAE |  RMSE |      R² |
| ---------------- | ----: | ----: | ------: |
| 24 hours         |  7.26 | 10.84 |  0.3316 |
| 48 hours         | 10.28 | 14.59 | -0.2153 |
| 72 hours         | 11.83 | 16.54 | -0.5544 |

The selected machine-learning models achieve lower MAE and RMSE than the persistence baseline across all three forecasting horizons. They also achieve higher R² scores.

This demonstrates that the trained models provide predictive value beyond simply assuming that future AQI will remain unchanged.

---

## Automated Pipelines

The project contains **three automated pipelines**, each serving a different purpose.

### 1. Hourly Pipeline — Hopsworks

The hourly pipeline runs directly on **Hopsworks** and is responsible for continuously updating the AQI feature data.

It:

1. Fetches the latest environmental data.
2. Processes the incoming data.
3. Performs feature engineering.
4. Stores the updated features in the Hopsworks Feature Store.

External APIs
      ↓
Hourly Pipeline
      ↓
Feature Engineering
      ↓
Hopsworks Feature Store


This pipeline provides continuously updated data for predictions and model training.

---

### 2. Daily Model Training — Hopsworks

The daily model training pipeline runs on **Hopsworks** and retrains the forecasting models using the latest available historical feature data.

It:

1. Retrieves historical features.
2. Creates 24h, 48h and 72h forecasting targets.
3. Trains the selected machine-learning models.
4. Evaluates model performance.
5. Registers the updated models in the Hopsworks Model Registry.

Hopsworks Feature Store
          ↓
    Training Dataset
          ↓
      Model Training
          ↓
       Evaluation
          ↓
   Hopsworks Model Registry


This allows the forecasting models to be periodically updated as new environmental data becomes available.

---

### 3. AQI Feature Pipeline — GitHub Actions

The **AQI Feature Pipeline** is automated using **GitHub Actions** and runs every hour.

It:

1. Retrieves environmental data.
2. Runs the feature pipeline.
3. Performs feature processing.
4. Updates the processed AQI feature data in Hopsworks.

GitHub Actions
      ↓
AQI Feature Pipeline
      ↓
Feature Processing
      ↓
Hopsworks


The workflow can also be triggered manually using GitHub Actions.

Together, these three pipelines provide automation across data collection, feature processing, storage, prediction, and model retraining.

---

## Streamlit Dashboard

The project includes an interactive web dashboard built with **Streamlit**.

### Current AQI

Displays the latest AQI value and its corresponding AQI category.

### AQI Forecasts

Displays AQI forecasts for:

* +24 hours
* +48 hours
* +72 hours

### Pollutant Monitoring

Displays current pollutant levels including:

* PM2.5
* PM10
* Ozone
* CO
* NO₂
* SO₂

### EDA & Trends

The dashboard includes exploratory analysis such as:

* AQI trends
* AQI distribution
* AQI boxplot
* Average AQI by hour
* Average AQI by month
* Pollutant trends
* AQI vs pollutant relationships
* Correlation analysis

### Prediction Explainability

SHAP is used to explain the 24-hour AQI prediction.

The dashboard displays:

* Global feature importance
* Top features influencing the latest prediction
* Positive and negative feature contributions
* SHAP contribution values

### AQI Alerts

The dashboard uses the EPA-style AQI scale to identify unhealthy and hazardous air-quality conditions and display alerts when AQI levels reach concerning categories.

Alerts can also indicate when a future forecast is expected to reach an unhealthy AQI category.

### Model Performance

The dashboard dynamically displays the performance of the forecasting models, including:

* MAE
* RMSE
* R²
* Forecast horizon
* Model type

---

## Project Structure

AQI_Quality_Index/
│
├── dashboard/
│   ├── app.py
│
├── src/
│   ├── config.py
│   ├── main.py
│   |
|   |── api/
│   |  └── fetch_air_data.py
|   |  └── fetch_weather.py
|   |  └── historical_air_data.py
|   |  └── historical_weather_data.py
|   |
│   ├── new_features/
│   │   └── feature_engineering.py
|   |
│   ├── pipeline/
│   │   └── backfill_pipeline.py
│   │   └── feature_pipeline.py
│   │   └── historical_pipeline.py
│   │   └── hourly_pipeline.py
│   │
│   ├── storage/
│   │   └── hopsworks_storage.py
|   |
│   ├── training/
│   │   └── train_model.py
│   │
│   └── prediction/
│       └── predict.py
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       └── daily_training.yml
│
├── models/
│   └── ...
│
├── requirements.txt
├── .gitignore
└── README.md

---

## Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* SHAP

### Data & Model Management

* Hopsworks Feature Store
* Hopsworks Model Registry

### Visualization

* Plotly
* Streamlit

### Automation

* GitHub Actions
* Hopsworks Jobs

### APIs

* OpenWeather
* Open-Meteo

---

## Dataset

The project uses historical environmental data for model development.

The final feature dataset contains approximately:

17,543 records
21 machine-learning features

Historical data is processed into a time-series dataset and used to create future AQI targets for:

AQI +24 hours
AQI +48 hours
AQI +72 hours

A time-aware train/test split is used so that future observations are not used to train models for earlier observations.

---

## Environment Variables

API keys and Hopsworks credentials are stored as environment variables rather than being committed to GitHub.

Required variables include:
OPENWEATHER_API_KEY
HOPSWORKS_API_KEY
HOPSWORKS_PROJECT
ENABLE_HOPSWORKS

For local development, these can be stored in a `.env` file.

Example:
OPENWEATHER_API_KEY=your_api_key
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT=your_project_name

**Do not commit ****`.env`**** or API keys to GitHub.**

---

## Running Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AQI_Quality_Index
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file and add the required API keys and Hopsworks credentials.

### 5. Run the feature pipeline

```bash
python -m src.main
```

### 6. Run model training

```bash
python -m src.training.train_model
```

### 7. Run predictions

```bash
python -m src.prediction.predict
```

### 8. Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

---

## Deployment

The Streamlit dashboard is deployed online using **Streamlit Community Cloud**.

The deployed dashboard connects to the Hopsworks Feature Store and Model Registry to retrieve current feature data and registered forecasting models.

This allows users to access:

* Current AQI
* Forecasted AQI
* AQI alerts
* Historical trends
* Pollutant information
* Model performance
* SHAP explanations


without running the application locally.

---

## End-to-End Workflow

The final system operates as follows:

                    External APIs
                         │
                         ▼
               ┌──────────────────┐
               │ Data Collection  │
               └────────┬─────────┘
                        │
                        ▼
               Feature Engineering
                        │
                        ▼
             Hopsworks Feature Store
                    │           │
                    │           │
             ┌──────▼─────┐ ┌──▼───────────┐
             │   Model    │ │  Prediction  │
             │  Training  │ │   Pipeline   │
             └──────┬─────┘ └──────┬───────┘
                    │              │
                    ▼              ▼
             Model Registry    AQI Forecast
                    │              │
                    └──────┬───────┘
                           ▼
                  Streamlit Dashboard

The automated system continuously updates environmental data, processes new features, retrains forecasting models, and provides updated AQI predictions through the dashboard.

---

## Project Objectives Achieved

* [x] End-to-end AQI prediction system
* [x] Automated hourly AQI pipeline using Hopsworks
* [x] Automated daily model training using Hopsworks
* [x] Automated AQI feature pipeline using GitHub Actions
* [x] Historical data backfill
* [x] Feature engineering
* [x] Hopsworks Feature Store integration
* [x] Multiple forecasting models
* [x] Persistence baseline evaluation
* [x] Model evaluation using MAE, RMSE and R²
* [x] Hopsworks Model Registry
* [x] 24-hour, 48-hour and 72-hour AQI forecasts
* [x] Interactive Streamlit dashboard
* [x] Exploratory Data Analysis
* [x] SHAP-based model explainability
* [x] AQI alerts
* [x] Dynamic model performance monitoring
* [x] Cloud deployment

---

## Internship Project

This project was developed as part of my **Data Science Internship at 10 Pearls**.

The project provided practical experience in:

* Data collection
* Data preprocessing
* Feature engineering
* Time-series forecasting
* Machine learning
* Model evaluation
* Feature stores
* Model registries
* MLOps
* CI/CD automation
* Cloud deployment
* Data visualization
* Model explainability

---

## Future Improvements

Potential future improvements include:

* Adding additional forecasting models such as LSTM/GRU
* Incorporating more historical and meteorological variables
* Improving long-horizon forecasting performance
* Adding automated notifications through email or messaging platforms
* Supporting additional cities
* Adding model drift monitoring

---

## License

This project was developed for educational and internship purposes.
