"""
src/data_processing.py
----------------------
Handles:
  1. Data validation and quality checks
  2. Preprocessing pipeline (encoding, scaling, imputation)
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os

# -- Reproducibility -----------------------------------------------------------
SEED = 42

# ==============================================================================
# 1. DATA VALIDATION
# ==============================================================================

def validate_data(df: pd.DataFrame) -> dict:
    """
    Run basic data quality checks and return a summary report.

    Checks:
      - Shape and column completeness
      - Missing value counts
      - Target class balance
      - Numeric range sanity (age, salary, tenure)

    Returns:
        dict with validation results
    """
    report = {}

    # Shape
    report["rows"], report["cols"] = df.shape

    # Missing values
    missing = df.isnull().sum()
    report["missing"] = missing[missing > 0].to_dict()

    # Class balance
    if "enrolled" in df.columns:
        vc = df["enrolled"].value_counts(normalize=True)
        report["class_balance"] = {int(k): round(float(v), 3) for k, v in vc.items()}

    # Numeric sanity
    if "age" in df.columns:
        report["age_range"] = (int(df["age"].min()), int(df["age"].max()))
    if "salary" in df.columns:
        report["salary_range"] = (float(df["salary"].min()), float(df["salary"].max()))

    print("[validate] Data Quality Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    return report


# ==============================================================================
# 2. FEATURE ENGINEERING + PREPROCESSING PIPELINE
# ==============================================================================

# Column definitions
NUMERIC_COLS     = ["age", "salary", "tenure_years"]
CATEGORICAL_COLS = ["gender", "marital_status", "employment_type", "region", "has_dependents"]
DROP_COLS        = ["employee_id"]  # identifier - not predictive
TARGET_COL       = "enrolled"


def build_preprocessor() -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that:
      - Imputes missing numeric values with the median (robust to outliers)
      - Scales numeric features with StandardScaler
      - Imputes missing categorical values with the most frequent value
      - OneHot-encodes categorical features (handle_unknown='ignore' for safety)

    Returns:
        sklearn ColumnTransformer (not yet fitted)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # median robust to salary outliers
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,     NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ], remainder="drop")  # explicitly drops employee_id and target

    return preprocessor


def load_and_split(filepath: str, test_size: float = 0.2):
    """
    Load CSV, drop identifier columns, split into X/y train/test.

    Args:
        filepath:  Path to employee_data.csv
        test_size: Fraction held out for testing (default 0.2)

    Returns:
        X_train, X_test, y_train, y_test as DataFrames/Series
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(filepath)
    print(f"[load] Loaded {len(df):,} rows from {filepath}")

    validate_data(df)

    # Drop non-feature columns
    X = df.drop(columns=DROP_COLS + [TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    print(f"[split] Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test
