def store_on_hopsworks(df):
    import hopsworks
    from src.config import hopsworks_api_key, hopsworks_project

    project = hopsworks.login(
        project = hopsworks_project,
        api_key_value = hopsworks_api_key
    )

    feature_store = project.get_feature_store()

    feature_group = feature_store.get_or_create_feature_group(
        name="historical_aqi_features",
        version=1,
        description="Historical AQI dataset with engineered features",
        primary_key=["time"],
        event_time="time"
    )

    feature_group.insert(
        df,
        wait=True
    )

    print("Successfully uploaded to Hopsworks!")


def get_previous_features():
    import hopsworks
    from datetime import datetime, timedelta, timezone
    from src.config import hopsworks_api_key, hopsworks_project

    project = hopsworks.login(
        project=hopsworks_project,
        api_key_value=hopsworks_api_key
    )

    feature_store = project.get_feature_store()

    feature_group = feature_store.get_feature_group(
        name="historical_aqi_features",
        version=1
    )

    # Read only the last 7 days instead of the entire Feature Group
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)

    df = feature_group.read(
        start_time=start_time,
        end_time=end_time,
        dataframe_type="pandas",
        read_options={
            "arrow_flight_config": {
                "timeout": 900
            }
        }
    )

    if df.empty:
        raise ValueError("No recent historical features found in Hopsworks.")

    df = df.sort_values("time")

    return df.tail(2)