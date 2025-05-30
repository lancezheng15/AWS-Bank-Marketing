import os
import boto3
import pandas as pd
from dotenv import load_dotenv
from utils.logger import setup_logger
from config import AWS_CONFIG, MODEL_CONFIG
from botocore.exceptions import ClientError

# Initialize logger
logger = setup_logger(__name__)

# Load environment variables
load_dotenv()

def test_aws_connectivity():
    """Test AWS connectivity and S3 bucket access"""
    try:
        # Create S3 client
        s3 = boto3.client('s3',
                         region_name=AWS_CONFIG['region'],
                         aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                         aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
        
        # Test bucket access
        bucket_name = AWS_CONFIG['bucket_name']
        s3.head_bucket(Bucket=bucket_name)
        logger.info(f"✓ Successfully connected to S3 bucket: {bucket_name}")
        
        # List objects in bucket
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            logger.info(f"✓ Found {len(response['Contents'])} objects in bucket")
            for obj in response['Contents'][:5]:  # Show first 5 objects
                logger.info(f"  - {obj['Key']}")
        
        return True
    except ClientError as e:
        logger.error(f"✗ AWS S3 Error: {str(e)}")
        return False

def test_model_files():
    """Test if all required model files are present in S3"""
    try:
        s3 = boto3.client('s3')
        bucket_name = AWS_CONFIG['bucket_name']
        
        for model_name, config in MODEL_CONFIG.items():
            model_path = os.path.join(AWS_CONFIG['models_path'], config['file_name'])
            try:
                s3.head_object(Bucket=bucket_name, Key=model_path)
                logger.info(f"✓ Model file found: {model_path}")
            except ClientError:
                logger.error(f"✗ Model file not found: {model_path}")
        
        return True
    except ClientError as e:
        logger.error(f"✗ Error checking model files: {str(e)}")
        return False

def test_data_access():
    """Test if training data is accessible"""
    try:
        s3 = boto3.client('s3')
        bucket_name = AWS_CONFIG['bucket_name']
        data_path = AWS_CONFIG['data_path']
        
        try:
            # Check if data file exists
            s3.head_object(Bucket=bucket_name, Key=data_path)
            logger.info(f"✓ Data file found: {data_path}")
            
            # Try to read first few rows
            obj = s3.get_object(Bucket=bucket_name, Key=data_path)
            df = pd.read_csv(obj['Body'], nrows=5)
            logger.info(f"✓ Successfully read data sample with shape: {df.shape}")
            logger.info(f"✓ Columns found: {', '.join(df.columns)}")
            
            return True
        except ClientError:
            logger.error(f"✗ Data file not found: {data_path}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error accessing data: {str(e)}")
        return False

def main():
    """Run all configuration tests"""
    logger.info("Starting configuration tests...")
    
    # Test AWS connectivity
    logger.info("\n1. Testing AWS Connectivity:")
    aws_test = test_aws_connectivity()
    
    # Test model files
    logger.info("\n2. Testing Model Files:")
    models_test = test_model_files()
    
    # Test data access
    logger.info("\n3. Testing Data Access:")
    data_test = test_data_access()
    
    # Summary
    logger.info("\nTest Summary:")
    logger.info(f"AWS Connectivity: {'✓' if aws_test else '✗'}")
    logger.info(f"Model Files: {'✓' if models_test else '✗'}")
    logger.info(f"Data Access: {'✓' if data_test else '✗'}")
    
    if aws_test and models_test and data_test:
        logger.info("\n✓ All configuration tests passed!")
    else:
        logger.error("\n✗ Some tests failed. Please check the logs above for details.")

if __name__ == "__main__":
    main() 