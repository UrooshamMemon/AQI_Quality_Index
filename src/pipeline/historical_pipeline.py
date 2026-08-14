from src.pipeline.backfill_pipelinie import merge_historical_data
from src.new_features.feature_engineering import engineer_historical_features

def run_historical_pipeline():
    history_df = merge_historical_data()
    history_df = engineer_historical_features(history_df)

    return history_df