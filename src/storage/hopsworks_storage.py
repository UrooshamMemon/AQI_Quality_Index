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

    feature_group.insert(df, wait=True)

    print("Successfully uploaded to Hopsworks!")


def get_previous_features():
    import hopsworks
    from src.config import hopsworks_api_key, hopsworks_project
    
    project = hopsworks.login(
        project = hopsworks_project,
        api_key_value = hopsworks_api_key
    )
    
    feature_store = project.get_feature_store()

    feature_group = feature_store.get_feature_group(
        name="historical_aqi_features",
        version=1
    )

    df = feature_group.select_all().read()

    df = df.sort_values("time")

    return df.tail(2)
