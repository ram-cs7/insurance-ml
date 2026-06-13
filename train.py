"""
train.py
--------
Entry point for training the insurance enrollment prediction model.

Usage:
    python train.py              # train all models with hyperparameter tuning
    python train.py --no-tune    # train without tuning (faster, for testing)
"""

import argparse
import os

from src.data_processing import load_and_split
from src.model           import train_all_models, select_best_model
from src.evaluate        import full_evaluation_report

DATA_PATH = "data/employee_data.csv"


def main(tune: bool = True):
    print("=" * 60)
    print("Insurance Enrollment - ML Training Pipeline")
    print("=" * 60)

    # Step 1: Ensure data exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please ensure the provided file is placed there.")
    print(f"\n[step 1] Using dataset: {DATA_PATH}")

    # Step 2: Load + split
    print("\n[step 2] Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH, test_size=0.2)

    # Step 3: Train candidate models
    print("\n[step 3] Training models...")
    if tune:
        print("         (Hyperparameter tuning ENABLED - this may take a minute)")
    else:
        print("         (Hyperparameter tuning DISABLED)")
    results = train_all_models(X_train, X_test, y_train, y_test, tune=tune)

    # Step 4: Select best model
    print("\n[step 4] Selecting best model...")
    best_name, best_pipeline, best_metrics = select_best_model(results)

    # Step 5: Full evaluation report
    print("\n[step 5] Generating full evaluation report...")
    full_evaluation_report(best_pipeline, X_test, y_test, model_name=best_name)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE.")
    print(f"Best model '{best_name}' saved to models/best_model.pkl")
    print("Run `mlflow ui` to view experiment tracking details.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune",  action="store_true", help="Skip hyperparameter tuning")
    args = parser.parse_args()
    main(tune=not args.no_tune)
