# Bank Marketing ML Pipeline

This project implements a machine learning pipeline for predicting bank term deposit subscriptions. The pipeline includes data loading from S3, feature engineering, model training, and model deployment.

## Project Structure

```
.
├── data_loader.py         # Handles loading data from S3
├── feature_engineering.py # Handles data preprocessing and feature engineering
├── model_training.py      # Handles training and saving models
├── upload_models.py       # Handles uploading models to S3
├── pipeline.py           # Main pipeline orchestration
└── requirements.txt      # Project dependencies
```

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create and configure .env file:
```bash
# Create .env file with your AWS credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=
AWS_BUCKET_NAME=
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure AWS credentials:
- Ensure you have AWS credentials configured in `~/.aws/credentials` or via environment variables
- Required permissions: S3 read/write access

## Running the Pipeline

Run the complete pipeline:
```bash
python pipeline.py
```

This will:
1. Load data from S3
2. Preprocess the data
3. Train three models (Random Forest, Logistic Regression, XGBoost)
4. Upload models to S3

## Models

The pipeline trains and saves three models:
- Random Forest Classifier
- Logistic Regression
- XGBoost Classifier

Models are saved locally in the `models/` directory and then uploaded to S3.

## AWS Configuration

The project expects:
- S3 bucket: `cloud-team3`
- Input data path: `bank-additional/data/bank-additional-full.csv`
- Model storage path: `models/`

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Monitoring

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

1. Build the Docker image
2. Push to ECR
