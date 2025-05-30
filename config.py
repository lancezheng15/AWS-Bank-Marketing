import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS Configuration
AWS_CONFIG = {
    'bucket_name': os.getenv('AWS_BUCKET_NAME', 'cloud-team3'),
    'region': os.getenv('AWS_REGION', 'us-east-2'),
    'data_path': 'bank-additional/data/bank-additional-full.csv',
    'models_path': 'models/'
}

# Model Configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'file_name': 'random_forest.pkl'
    },
    'logistic_regression': {
        'random_state': 42,
        'max_iter': 1000,
        'file_name': 'logistic_regression.pkl'
    },
    'xgboost': {
        'random_state': 42,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'file_name': 'xgboost.pkl'
    }
}

# Data Processing Configuration
DATA_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'binary_features': ['default', 'housing', 'loan'],
    'categorical_features': [
        'job', 'marital', 'education', 'contact',
        'month', 'day_of_week', 'poutcome'
    ]
}

# Logging Configuration
LOG_CONFIG = {
    'log_file': 'logs/pipeline.log',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'level': 'INFO'
} 