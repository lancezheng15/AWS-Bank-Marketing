from data_loader import load_data_from_s3
from feature_engineering import preprocess_data
from model_training import train_models
from upload_models import upload_models_to_s3

def run_pipeline():
    """
    Run the complete pipeline from data loading to model upload
    """
    print("Starting pipeline...")
    
    # Step 1: Load data from S3
    print("\n1. Loading data from S3...")
    df = load_data_from_s3()
    if df is None:
        print("Failed to load data. Pipeline stopped.")
        return False
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    if X_train is None:
        print("Failed to preprocess data. Pipeline stopped.")
        return False
    
    # Step 3: Train models
    print("\n3. Training models...")
    training_success = train_models(X_train, X_test, y_train, y_test)
    if not training_success:
        print("Failed to train models. Pipeline stopped.")
        return False
    
    # Step 4: Upload models to S3
    print("\n4. Uploading models to S3...")
    upload_models_to_s3()
    
    print("\nPipeline completed successfully!")
    return True

if __name__ == "__main__":
    run_pipeline() 