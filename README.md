# Bank Marketing ML Pipeline

This project implements a machine learning pipeline for predicting bank marketing campaign success using AWS cloud services.

## Project Overview

The project uses a bank marketing dataset to predict whether a client will subscribe to a term deposit. It implements a complete ML pipeline from data preprocessing to model deployment.

## Architecture

The system is built using the following components:
- **Data Storage**: AWS S3 for storing raw data and trained models
- **Model Training**: Multiple ML models (Random Forest, XGBoost, Logistic Regression)
- **Deployment**: Docker containerization with AWS ECS
- **Monitoring**: Custom logging and performance tracking

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── config.py             # Configuration management
├── data_loader.py        # S3 data loading utilities
├── feature_engineering.py # Data preprocessing
├── model_training.py     # Model training scripts
├── pipeline.py           # ML pipeline orchestration
├── upload_models.py      # Model upload to S3
├── utils/
│   ├── logger.py        # Logging utilities
│   └── monitoring.py    # Performance monitoring
├── models/              # Local model storage
├── requirements.txt     # Python dependencies
└── Dockerfile          # Container configuration
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/lancezheng15/AWS-Bank-Marketing.git
cd AWS-Bank-Marketing
```

2. Create and configure .env file:
```bash
# Create .env file with your AWS credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-2
AWS_BUCKET_NAME=cloud-team3
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses several configuration files:
- `config.py`: Central configuration for AWS, models, and logging
- `.env`: Environment variables for sensitive credentials
- `Dockerfile`: Container configuration

## Models

Three machine learning models are implemented:
1. Random Forest Classifier
2. XGBoost Classifier
3. Logistic Regression

Models are automatically saved to both local storage and S3.

## Data Pipeline

1. **Data Loading**: Raw data is loaded from S3
2. **Preprocessing**:
   - Binary feature encoding
   - Categorical feature one-hot encoding
   - Train-test splitting
3. **Model Training**: Multiple models are trained and evaluated
4. **Model Storage**: Trained models are saved to S3

## Monitoring and Logging

The project implements comprehensive monitoring:
- Model performance metrics
- Training time tracking
- Data statistics logging
- Error handling and reporting

Logs are stored in:
- Console output
- File-based logging (`logs/pipeline.log`)

## Security

- AWS credentials are managed through environment variables
- Sensitive information is excluded from version control
- Docker container security best practices
- Proper error handling and input validation

## Deployment

The application is containerized using Docker and deployed on AWS ECS:

1. Build the Docker image:
```bash
docker build -t bank-marketing-app .
```

2. Push to ECR:
```bash
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 528192012979.dkr.ecr.us-east-2.amazonaws.com
docker tag bank-marketing-app:latest 528192012979.dkr.ecr.us-east-2.amazonaws.com/bank-marketing-app:latest
docker push 528192012979.dkr.ecr.us-east-2.amazonaws.com/bank-marketing-app:latest
```

## Future Improvements

- Add AWS CloudWatch integration for monitoring
- Implement model A/B testing
- Add automated retraining pipeline
- Enhance API documentation
- Add unit tests

## Contributors

- Lance Zheng

## License

This project is licensed under the MIT License - see the LICENSE file for details. 