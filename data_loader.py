import boto3
import pandas as pd
from io import StringIO

def load_data_from_s3(bucket_name='cloud-team3', file_key='bank-additional/data/bank-additional-full.csv'):
    """
    Load data from S3 bucket
    """
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Download and decode
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        body = response['Body'].read().decode('utf-8')
        
        # Load into DataFrame
        df = pd.read_csv(StringIO(body), sep=';')
        print(f"Successfully loaded data from s3://{bucket_name}/{file_key}")
        return df
        
    except Exception as e:
        print(f"Error loading data from S3: {str(e)}")
        return None 