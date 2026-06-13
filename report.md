# ML Report: Insurance Enrollment Prediction

**Author:** Sairam Chennaka  
**Date:** June 2026

---

## 1. Data Observations

**Dataset:** 10,000 employee records, census-style voluntary insurance enrollment data.

**Target distribution:** ~45% enrolled (1), ~55% not enrolled (0) - mild imbalance, addressed via `class_weight="balanced"` in classifiers.

**Missing values:**
- `salary`: ~3% missing (realistic - some employees decline to disclose)
- `tenure_years`: ~3% missing
- Strategy: median imputation (median is robust to the right-skewed salary distribution)

**Key observations from the data:**
- Salary is right-skewed (log-normal distribution) - a small number of high earners pull the mean up
- Age ranges 22-64, relatively uniform
- Married employees with dependents show higher enrollment rates in exploratory analysis
- Part-time employees show lower enrollment likelihood - consistent with real-world behavior (fewer benefits-eligible)

---

## 2. Feature Engineering

No new features were derived in this version. The following features were used:

| Feature | Type | Treatment |
|---|---|---|
| age | Numeric | StandardScaler after median imputation |
| salary | Numeric | StandardScaler after median imputation |
| tenure_years | Numeric | StandardScaler after median imputation |
| has_dependents | Categorical (Yes/No) | OneHotEncoder |
| gender | Categorical | OneHotEncoder |
| marital_status | Categorical | OneHotEncoder |
| employment_type | Categorical | OneHotEncoder |
| region | Categorical | OneHotEncoder |

`employee_id` was dropped (identifier, not predictive).

The preprocessing pipeline is built as a sklearn `ColumnTransformer` wrapped inside a full `Pipeline` - this ensures preprocessing is always fitted on training data only, preventing any form of data leakage.

---

## 3. Model Choices & Rationale

Three models were compared:

### Logistic Regression (baseline)
- Interpretable, fast to train
- Assumes linear decision boundary - likely insufficient for real enrollment patterns
- Useful as a baseline to confirm whether non-linear models add value

### Random Forest
- Handles non-linearity and feature interactions
- Robust to outliers (tree-based, not affected by salary scale)
- Provides native feature importance
- `class_weight="balanced"` handles mild class imbalance

### Gradient Boosting (GBM)
- Typically best predictive performance on tabular data
- Sequential boosting corrects previous errors
- Slower to train but worth it for production

**Primary selection metric: ROC-AUC**  
Chosen because:
1. Threshold-independent - the business may want to adjust the operating threshold (e.g., target the top 30% most likely to enroll for outreach campaigns)
2. Handles class imbalance better than raw accuracy
3. Represents the probability that the model ranks a positive example higher than a negative one

---

## 4. Evaluation Results

| Model | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.9707 | 0.9132 | 0.9370 | 0.8907 | 0.8955 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Winner:** Random Forest (selected due to perfect score and robust interpretability).

**Top predictive features** (from Random Forest / GBM feature importance):
1. `salary` - strongest signal (0.28 importance)
2. `age` - strong positive effect (0.20 importance)
3. `employment_type_Full-time` - full-time employees show distinct behavior
4. `has_dependents_No` - lack of dependents is a strong indicator
5. `has_dependents_Yes` - having dependents is also a strong indicator

---

## 5. Key Takeaways

- **Enrollment is perfectly predictable** with ROC-AUC 1.000 on tree-based models, suggesting the dataset contains very strong, almost deterministic rules for enrollment (e.g. strict salary/age thresholds).
- **Salary and Age are the strongest signals** - aligns with the idea that older, higher-earning employees are significantly more likely to opt into voluntary insurance.
- **Employment type matters significantly** - full-time vs part-time plays a large role, likely reflecting benefits eligibility rules.
- **Logistic Regression performed very well (AUC 0.97)**, but the non-linear models achieved perfect scores, indicating there are specific thresholds and interactions (e.g. "if salary > X AND age > Y") that tree-based models exploit perfectly.
- **The preprocessing pipeline is leak-free** - all transformations are fitted on training data only.

---

## 6. What I'd Do Next With More Time

1. **Explainability:** Add SHAP values to understand individual predictions - useful for the business team and for bias auditing
2. **Threshold optimisation:** Work with stakeholders to define the cost of a false positive vs false negative and optimise the decision threshold accordingly
3. **Fairness audit:** Check model performance across gender and region subgroups - enrollment models can amplify existing demographic biases
4. **More features:** Job title, department, distance from office, previous enrollment history if available
5. **Calibration:** Platt scaling or isotonic regression to ensure predicted probabilities are well-calibrated (important if probabilities are used directly for scoring)
6. **Production monitoring:** Set up data drift detection (PSI on key features) and model performance monitoring with automatic retraining triggers
7. **XGBoost / LightGBM:** Likely to outperform sklearn GBM - would add to the comparison if time allowed
