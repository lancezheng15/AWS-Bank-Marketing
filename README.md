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

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials:
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