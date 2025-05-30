import time
from functools import wraps
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from utils.logger import setup_logger

logger = setup_logger('monitoring')

def monitor_time(func):
    """Decorator to monitor function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} execution time: {execution_time:.2f} seconds")
        return result
    return wrapper

def monitor_model_performance(y_true, y_pred, model_name):
    """
    Monitor and log model performance metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    logger.info(f"Performance metrics for {model_name}:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.3f}")
    
    return metrics

def monitor_data_statistics(df, stage):
    """
    Monitor and log data statistics
    """
    logger.info(f"Data statistics at {stage}:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    
    if df.select_dtypes(include=[np.number]).columns.any():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        logger.info("Numeric columns statistics:")
        logger.info(df[numeric_cols].describe().to_string())
    
    return {
        'shape': df.shape,
        'missing_values': df.isnull().sum().sum(),
        'numeric_stats': df.describe().to_dict() if df.select_dtypes(include=[np.number]).columns.any() else None
    } 