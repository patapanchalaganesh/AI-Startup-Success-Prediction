import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_startup_model():
    print("--- Step 2: Training Machine Learning Model ---")
    data_path = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
    model_dir = os.path.join(BASE_DIR, 'models')

    if not os.path.exists(data_path):
        from src.data_preprocessing import preprocess_startup_data
        df = preprocess_startup_data()
    else:
        df = pd.read_csv(data_path)

    os.makedirs(model_dir, exist_ok=True)

    X = df[['Amount_USD', 'Clean_City', 'Clean_Industry', 'Clean_Investment_Type', 'Investor_Score']]
    y = df['Startup_Status']

    categorical_cols = ['Clean_City', 'Clean_Industry', 'Clean_Investment_Type']
    numerical_cols = ['Amount_USD', 'Investor_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_prep, y_train)

    # Save artifacts into models/
    joblib.dump(model, os.path.join(model_dir, 'user_best_model.pkl'))
    joblib.dump(preprocessor, os.path.join(model_dir, 'user_preprocessor.pkl'))

    print(f"Model trained and saved successfully to '{model_dir}'!")
    return model, preprocessor

if __name__ == "__main__":
    train_startup_model()