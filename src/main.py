from src.pipeline.hourly_pipeline import run_hourly_pipeline
from src.pipeline.hourly_pipeline import save_local_features
from src.storage.hopsworks_storage import store_on_hopsworks
from src.config import ENABLE_HOPSWORKS

def main():
    current_df = run_hourly_pipeline()

    print("\nCurrent Hourly Features:")
    print(current_df)   
    
    if ENABLE_HOPSWORKS:
        print("\nFINAL COLUMNS:")
        print(current_df.columns.tolist()) 
        store_on_hopsworks(current_df)
    else:
        save_local_features(current_df)
        print("Hopsworks disabled. Running on local server.")

if __name__ == "__main__":
    main()