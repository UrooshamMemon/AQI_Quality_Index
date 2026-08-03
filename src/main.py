from src.pipeline.feature_pipeline import create_features
from src.pipeline.backfill_pipelinie import merge_historical_data
from src.new_features.feature_engineering import engineering_features

def main():
    features = create_features()
    history_df = merge_historical_data()

    history_df = engineering_features(history_df)

    print("Current Feature Vector")
    print(features)

    print("\nHistorical Dataset")
    print(f"Total Records: {len(history_df)}")

    print("\nFirst 5 Records:")
    print(history_df.head())

if __name__ == "__main__":
    main()