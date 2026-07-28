import os
import joblib
import pandas as pd
from src.data_preprocessing import preprocess_startup_data
from src.train_model import train_startup_model
from src.evaluate_model import evaluate_startup_model


def run_all_tests():
    print("=== Running Integration & System Tests ===")

    # 1. Test Preprocessing
    df = preprocess_startup_data()
    assert os.path.exists('data/processed_data.csv'), "processed_data.csv missing!"
    print("Test 1 Passed: Data Preprocessing complete.")

    # 2. Test Training
    model, preprocessor = train_startup_model()
    assert os.path.exists('models/user_best_model.pkl'), "best_model.pkl missing!"
    assert os.path.exists('models/user_preprocessor.pkl'), "preprocessor.pkl missing!"
    print("Test 2 Passed: Model Training complete.")

    # 3. Test Evaluation
    evaluate_startup_model()
    print("Test 3 Passed: Model Evaluation complete.")

    print("=== ALL TESTS PASSED SUCCESSFULLY! ===")


if __name__ == '__main__':
    run_all_tests()