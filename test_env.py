import os
from dotenv import load_dotenv
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Load environment variables
load_dotenv()

def test_env_variables():
    # Check if environment variables are set (without printing the actual values)
    aws_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'AWS_BUCKET_NAME']
    
    for var in aws_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"{var} is set ✓")
            # For AWS credentials, only show the first and last 4 characters
            if var in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'] and value:
                masked_value = value[:4] + '*' * (len(value)-8) + value[-4:]
                logger.info(f"{var} value: {masked_value}")
        else:
            logger.error(f"{var} is not set ✗")

if __name__ == "__main__":
    test_env_variables() 