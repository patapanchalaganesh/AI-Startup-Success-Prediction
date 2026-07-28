import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def evaluate_startup_model(data_path='data/processed_data.csv', model_dir='models'):
    print("--- Step 3: Evaluating Machine Learning Model ---")
    model_path = os.path.join(model_dir, 'user_best_model.pkl')
    prep_path = os.path.join(model_dir, 'user_preprocessor.pkl')

    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        from src.train_model import train_startup_model
        model, preprocessor = train_startup_model(data_path=data_path, model_dir=model_dir)
    else:
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)

    df = pd.read_csv(data_path)
    X = df[['Amount_USD', 'Clean_City', 'Clean_Industry', 'Clean_Investment_Type', 'Investor_Score']]
    y = df['Startup_Status']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_prep = preprocessor.transform(X_test)

    y_pred = model.predict(X_test_prep)
    y_proba = model.predict_proba(X_test_prep)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Model Evaluation Summary ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("--------------------------------\n")


if __name__ == "__main__":
    evaluate_startup_model()