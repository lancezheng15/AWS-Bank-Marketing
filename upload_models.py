import boto3
import os

def upload_models_to_s3():
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # S3 bucket and folder configuration
    bucket_name = 'cloud-team3'
    s3_folder = 'models/'
    
    # Local models directory
    local_models_dir = 'models'
    
    # Models to upload
    model_files = [
        'xgboost.pkl',
        'random_forest.pkl',
        'logistic_regression.pkl'
    ]
    
    # Upload each model
    for model_file in model_files:
        local_file_path = os.path.join(local_models_dir, model_file)
        s3_file_path = s3_folder + model_file
        
        try:
            print(f"Uploading {model_file} to s3://{bucket_name}/{s3_file_path}")
            s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
            print(f"Successfully uploaded {model_file}")
        except Exception as e:
            print(f"Error uploading {model_file}: {str(e)}")

if __name__ == "__main__":
    upload_models_to_s3() 