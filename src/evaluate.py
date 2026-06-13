"""
src/evaluate.py
---------------
Post-training analysis:
  1. Feature importance extraction
  2. Classification report printing
  3. Threshold analysis (business-relevant)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.data_processing import NUMERIC_COLS, CATEGORICAL_COLS


def get_feature_names(pipeline) -> list:
    """
    Extract human-readable feature names after OneHotEncoding.
    Works with sklearn's ColumnTransformer.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder  = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    return NUMERIC_COLS + cat_features


def print_feature_importance(pipeline, model_name: str, top_n: int = 15):
    """
    Print top N most important features.
    Works for tree-based models (RandomForest, GradientBoosting).
    For LogisticRegression, uses absolute coefficient values.
    """
    classifier = pipeline.named_steps["classifier"]
    feat_names = get_feature_names(pipeline)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        label = "Feature Importance"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        label = "Abs. Coefficient"
    else:
        print(f"  [importance] Not available for {model_name}")
        return

    fi_df = pd.DataFrame({
        "feature":    feat_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    print(f"\n[importance] Top {top_n} features for {model_name} ({label}):")
    print(fi_df.to_string(index=False))
    return fi_df


def full_evaluation_report(pipeline, X_test, y_test, model_name: str):
    """
    Print a comprehensive evaluation report including:
      - Classification report (precision/recall/F1 per class)
      - Confusion matrix
      - ROC-AUC
      - Threshold analysis (what happens at 0.3/0.4/0.5/0.6 threshold)
    """
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT — {model_name.upper()}")
    print(f"{'='*60}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Enrolled", "Enrolled"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    print(f"\nROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")

    # Threshold analysis — useful for business to decide operating point
    print("\nThreshold Analysis (effect on precision/recall trade-off):")
    print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'% Flagged'}")
    from sklearn.metrics import precision_score, recall_score, f1_score
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_t = (y_pred_prob >= t).astype(int)
        p = precision_score(y_test, preds_t, zero_division=0)
        r = recall_score(y_test, preds_t, zero_division=0)
        f = f1_score(y_test, preds_t, zero_division=0)
        flagged = preds_t.mean() * 100
        print(f"  {t:<12.1f} {p:<12.3f} {r:<12.3f} {f:<10.3f} {flagged:.1f}%")

    print_feature_importance(pipeline, model_name)
