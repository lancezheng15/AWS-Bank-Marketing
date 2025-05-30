import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
import os

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and save them
    """
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Initialize models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42)
        }
        
        # Train and save each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Save model
            model_path = f"models/{name}.pkl"
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")
            
            # Calculate and print accuracy
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            print(f"{name} - Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error in training models: {str(e)}")
        return False 