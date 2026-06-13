"""
src/model.py
------------
Handles:
  1. Model definitions (Logistic Regression, Random Forest, Gradient Boosting)
  2. Training with MLflow experiment tracking
  3. Hyperparameter tuning via RandomizedSearchCV
  4. Model persistence (save / load)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics         import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, classification_report
)
from src.data_processing import build_preprocessor

# -- Paths ---------------------------------------------------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -- MLflow experiment name ----------------------------------------------------
EXPERIMENT_NAME = "insurance_enrollment"


# ==============================================================================
# 1. MODEL REGISTRY
# Model configs: name -> (estimator, hyperparameter search space)
# ==============================================================================

def get_model_configs() -> dict:
    """
    Return candidate models with their hyperparameter search spaces.

    We compare three models:
      - LogisticRegression: interpretable baseline; good for linear decision boundaries
      - RandomForest: handles non-linearity, robust to outliers, provides feature importance
      - GradientBoosting: typically best performance; slower to train
    """
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "param_dist": {
                "classifier__C":       [0.01, 0.1, 1.0, 10.0],
                "classifier__penalty": ["l1", "l2"],
                "classifier__solver":  ["liblinear", "saga"],
            }
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
            "param_dist": {
                "classifier__n_estimators":      [100, 200, 300],
                "classifier__max_depth":         [None, 5, 10, 20],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__max_features":      ["sqrt", "log2"],
            }
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "param_dist": {
                "classifier__n_estimators":  [100, 200],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__max_depth":     [3, 5, 7],
                "classifier__subsample":     [0.8, 1.0],
            }
        },
    }


# ==============================================================================
# 2. TRAIN + EVALUATE SINGLE MODEL
# ==============================================================================

def build_full_pipeline(estimator) -> Pipeline:
    """
    Wrap preprocessor + classifier into a single sklearn Pipeline.
    This ensures preprocessing is always fitted on train data only — prevents leakage.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   estimator),
    ])


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """
    Compute evaluation metrics on held-out test set.

    Metrics chosen:
      - ROC-AUC: primary metric (handles class imbalance better than accuracy)
      - F1:      balances precision and recall (important for voluntary enrollment)
      - Precision/Recall: separately to understand false positive vs false negative trade-off
      - Accuracy: reported but not used for model selection

    Args:
        pipeline: fitted sklearn Pipeline
        X_test, y_test: held-out test data

    Returns:
        dict of metric name -> float
    """
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc":   round(roc_auc_score(y_test, y_pred_prob), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
    }
    return metrics


# ==============================================================================
# 3. TRAIN ALL MODELS WITH MLFLOW TRACKING
# ==============================================================================

def train_all_models(X_train, X_test, y_train, y_test, tune: bool = True) -> dict:
    """
    Train all candidate models, optionally tune hyperparameters, log to MLflow.

    Args:
        X_train, X_test, y_train, y_test: split data
        tune: whether to run RandomizedSearchCV (slower but better results)

    Returns:
        dict mapping model_name -> {"pipeline": ..., "metrics": ..., "best_params": ...}
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    configs  = get_model_configs()
    results  = {}
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, config in configs.items():
        print(f"\n[train] -- {model_name} ------------------------------")

        with mlflow.start_run(run_name=model_name):
            pipeline = build_full_pipeline(config["estimator"])

            # -- Hyperparameter tuning -----------------------------------------
            if tune and config.get("param_dist"):
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=config["param_dist"],
                    n_iter=10,                # 10 random configs - balanced speed vs quality
                    cv=cv,
                    scoring="roc_auc",        # optimise for AUC
                    random_state=42,
                    n_jobs=-1,
                    verbose=0,
                )
                search.fit(X_train, y_train)
                best_pipeline   = search.best_estimator_
                best_params     = search.best_params_
                cv_auc          = round(search.best_score_, 4)
                print(f"  Best CV AUC: {cv_auc} | Params: {best_params}")
            else:
                pipeline.fit(X_train, y_train)
                best_pipeline = pipeline
                best_params   = {}
                cv_auc        = None

            # -- Evaluate on held-out test set ---------------------------------
            metrics = evaluate_model(best_pipeline, X_test, y_test)
            print(f"  Test metrics: {metrics}")

            # -- Log to MLflow --------------------------------------------------
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            if cv_auc:
                mlflow.log_metric("cv_roc_auc", cv_auc)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("tuned", tune)
            mlflow.sklearn.log_model(best_pipeline, artifact_path=model_name)

            # -- Save locally ---------------------------------------------------
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(best_pipeline, f)

            results[model_name] = {
                "pipeline":    best_pipeline,
                "metrics":     metrics,
                "best_params": best_params,
                "cv_auc":      cv_auc,
            }

    return results


# ==============================================================================
# 4. MODEL SELECTION
# ==============================================================================

def select_best_model(results: dict) -> tuple:
    """
    Select the best model by test ROC-AUC.

    ROC-AUC is chosen as the primary metric because:
      - The enrollment dataset may be imbalanced
      - AUC is threshold-independent - useful when the business may want to
        adjust the decision threshold later (e.g., target top 30% likely enrollees)

    Returns:
        (best_model_name, best_pipeline, best_metrics)
    """
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best      = results[best_name]
    print(f"\n[select] Best model: {best_name} (AUC={best['metrics']['roc_auc']})")

    # Save best model separately for API serving
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best["pipeline"], f)

    # Save metrics summary as JSON for reporting
    summary = {name: res["metrics"] for name, res in results.items()}
    with open(os.path.join(MODEL_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return best_name, best["pipeline"], best["metrics"]


# ==============================================================================
# 5. MODEL LOADING
# ==============================================================================

def load_best_model():
    """Load the saved best model pipeline from disk."""
    path = os.path.join(MODEL_DIR, "best_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model found at {path}. Run train.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)
