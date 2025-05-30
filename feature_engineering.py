import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Preprocess the data with feature engineering
    """
    try:
        # Create a copy of the dataframe
        df = df.copy()
        
        # Convert target variable to binary
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
        
        # Convert binary categorical variables
        binary_features = ['default', 'housing', 'loan']
        for feature in binary_features:
            df[feature] = df[feature].map({'yes': 1, 'no': 0, 'unknown': -1})
        
        # One-hot encode categorical variables
        categorical_features = ['job', 'marital', 'education', 'contact', 
                              'month', 'day_of_week', 'poutcome']
        df = pd.get_dummies(df, columns=categorical_features, drop_first=False)
        
        # Split features and target
        X = df.drop('y', axis=1)
        y = df['y']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Successfully preprocessed data")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error in preprocessing data: {str(e)}")
        return None, None, None, None 