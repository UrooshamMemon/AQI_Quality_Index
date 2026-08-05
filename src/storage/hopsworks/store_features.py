import hopsworks
from src.config import hopsworks_api_key, hopsworks_project

def store_on_hopsworks(df):
    project = hopsworks.login(
        project = hopsworks_project,
        api_key = hopsworks_api_key
    )

    feature_store = project.get_feature_store()

    print("Connected Successfully!")