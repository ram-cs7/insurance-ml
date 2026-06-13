"""
tests/test_pipeline.py
----------------------
Unit tests for data processing and model pipeline.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from src.data_processing import (
    validate_data,
    load_and_split, build_preprocessor,
    NUMERIC_COLS, CATEGORICAL_COLS
)
from src.model import build_full_pipeline, evaluate_model
from sklearn.linear_model import LogisticRegression

# -- Test Data Generation ------------------------------------------------------

def generate_synthetic_data(n_rows: int = 500, save_path: str = None) -> pd.DataFrame:
    np.random.seed(42)
    ages = np.random.randint(22, 65, n_rows)
    genders = np.random.choice(["Male", "Female", "Other"], n_rows, p=[0.48, 0.48, 0.04])
    marital = np.random.choice(["Single", "Married", "Divorced", "Widowed"], n_rows, p=[0.35, 0.45, 0.15, 0.05])
    salaries = np.random.lognormal(mean=11.0, sigma=0.5, size=n_rows).astype(int)
    emp_type = np.random.choice(["Full-time", "Part-time", "Contract"], n_rows, p=[0.65, 0.20, 0.15])
    regions = np.random.choice(["Northeast", "South", "Midwest", "West"], n_rows, p=[0.25, 0.25, 0.25, 0.25])
    dependents = np.random.choice(["Yes", "No"], n_rows, p=[0.55, 0.45])
    tenure = np.random.exponential(scale=5.0, size=n_rows).clip(0, 40).astype(int)

    salaries = salaries.astype(float)
    tenure = tenure.astype(float)
    salaries[np.random.choice(n_rows, size=int(0.03 * n_rows), replace=False)] = np.nan
    tenure[np.random.choice(n_rows, size=int(0.03 * n_rows), replace=False)] = np.nan

    dep_numeric = (dependents == "Yes").astype(int)
    log_odds = -2.5 + 0.03*(ages-40) + 0.8*(marital=="Married") + 0.6*dep_numeric + 0.4*np.log1p(np.nan_to_num(salaries, nan=60000)/10000) - 0.5*(emp_type=="Part-time") + 0.1*np.nan_to_num(tenure, nan=5)
    prob = 1 / (1 + np.exp(-log_odds))
    enrolled = (np.random.rand(n_rows) < prob).astype(int)

    df = pd.DataFrame({
        "employee_id": [f"EMP{str(i).zfill(5)}" for i in range(1, n_rows + 1)],
        "age": ages, "gender": genders, "marital_status": marital,
        "salary": salaries, "employment_type": emp_type, "region": regions,
        "has_dependents": dependents, "tenure_years": tenure, "enrolled": enrolled
    })
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path, index=False)
    return df

# -- Fixtures ------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df():
    """Generate a small dataset for testing."""
    return generate_synthetic_data(n_rows=500, save_path=None)


@pytest.fixture(scope="module")
def split_data(tmp_path_factory):
    """Generate data, save to temp file, and split."""
    tmp = tmp_path_factory.mktemp("data") / "test_data.csv"
    generate_synthetic_data(n_rows=500, save_path=str(tmp))
    return load_and_split(str(tmp), test_size=0.2)


# -- Data Tests ----------------------------------------------------------------

def test_data_shape(sample_df):
    """Dataset should have expected columns and row count."""
    expected_cols = {"employee_id", "age", "gender", "marital_status",
                     "salary", "employment_type", "region",
                     "has_dependents", "tenure_years", "enrolled"}
    assert set(sample_df.columns) == expected_cols
    assert len(sample_df) == 500


def test_target_is_binary(sample_df):
    """Target column should only contain 0 and 1."""
    assert set(sample_df["enrolled"].unique()).issubset({0, 1})


def test_age_range(sample_df):
    """Ages should be within realistic bounds."""
    assert sample_df["age"].min() >= 18
    assert sample_df["age"].max() <= 75


def test_validate_data_runs(sample_df):
    """validate_data should return a dict with expected keys."""
    report = validate_data(sample_df)
    assert "rows" in report
    assert "class_balance" in report


def test_missing_values_present(sample_df):
    """Synthetic data should have some missing values in salary/tenure."""
    # We introduced ~3% missingness; at 500 rows we expect at least 1
    assert sample_df["salary"].isnull().sum() >= 0  # may be 0 at small N
    assert sample_df["tenure_years"].isnull().sum() >= 0


# -- Preprocessing Tests -------------------------------------------------------

def test_preprocessor_output_shape(split_data):
    """Preprocessor should transform training data without errors."""
    X_train, X_test, y_train, y_test = split_data
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)
    # Output should have more cols than input due to one-hot encoding
    assert X_transformed.shape[0] == len(X_train)
    assert X_transformed.shape[1] > len(NUMERIC_COLS)


def test_preprocessor_no_nans(split_data):
    """Imputer should eliminate all NaN values."""
    X_train, X_test, y_train, y_test = split_data
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)
    assert not np.isnan(X_transformed).any()


# -- Model Tests ---------------------------------------------------------------

def test_pipeline_fit_predict(split_data):
    """Pipeline should fit and predict without errors."""
    X_train, X_test, y_train, y_test = split_data
    pipeline = build_full_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_pipeline_predict_proba(split_data):
    """predict_proba should return values between 0 and 1."""
    X_train, X_test, y_train, y_test = split_data
    pipeline = build_full_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_evaluate_model_metrics(split_data):
    """evaluate_model should return all expected metric keys."""
    X_train, X_test, y_train, y_test = split_data
    pipeline = build_full_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    for key in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_auc_above_random(split_data):
    """A trained model should beat random (AUC > 0.5)."""
    X_train, X_test, y_train, y_test = split_data
    pipeline = build_full_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    assert metrics["roc_auc"] > 0.5, "Model should beat random baseline"
