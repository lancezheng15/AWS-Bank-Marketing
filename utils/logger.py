import logging
import os
from config import LOG_CONFIG

def setup_logger(name):
    """
    Set up logger with specified configuration
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(LOG_CONFIG['level'])
    
    # Create formatter
    formatter = logging.Formatter(LOG_CONFIG['format'])
    
    # Create file handler
    file_handler = logging.FileHandler(LOG_CONFIG['log_file'])
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 