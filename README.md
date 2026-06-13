# Insurance Enrollment Predictor

ML pipeline to predict whether an employee will opt into a voluntary insurance product, based on demographic and employment data.

**Author:** Sairam Chennaka - github.com/ram-cs7

---

## Quick Start

```bash
git clone https://github.com/ram-cs7/insurance-ml.git
cd insurance-ml
pip install -r requirements.txt

# 1. Train models (trains + saves best model)
python train.py

# 2. Run tests
pytest tests/ -v

# 3. Start prediction API
uvicorn src.api:app --reload --port 8000

# 4. View MLflow experiment results
mlflow ui
# Open: http://localhost:5000
```

---

## Project Structure

```
insurance-ml/
|-- src/
|   |-- data_processing.py   # Data validation, preprocessing pipeline
|   |-- model.py             # Model training, tuning, MLflow tracking, saving
|   |-- evaluate.py          # Feature importance, full evaluation report
|   `-- api.py               # FastAPI prediction server
|-- tests/
|   `-- test_pipeline.py     # Unit tests (pytest)
|-- data/
|   `-- employee_data.csv    # Provided dataset
|-- models/
|   |-- best_model.pkl       # Best model saved after training
|   `-- metrics_summary.json # All model metrics
|-- train.py                 # Training entry point
|-- requirements.txt
|-- report.md
`-- README.md
```

---

## Training Options

```bash
python train.py              # Full training with hyperparameter tuning (recommended)
python train.py --no-tune    # Skip tuning (faster, for development)
```

---

## API Usage

After starting the server (`uvicorn src.api:app --reload`):

### Single prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "Female",
    "marital_status": "Married",
    "salary": 75000,
    "employment_type": "Full-time",
    "region": "Northeast",
    "has_dependents": "Yes",
    "tenure_years": 5
  }'
```

Response:
```json
{
  "enrolled_probability": 0.7823,
  "prediction": 1,
  "confidence": "High"
}
```

### Batch prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"age": 35, ...}, {"age": 28, ...}]'
```

### Interactive docs
Open http://localhost:8000/docs for Swagger UI.

---

## Valid Input Values

| Field | Valid Values |
|---|---|
| gender | Male, Female, Other |
| marital_status | Single, Married, Divorced, Widowed |
| employment_type | Full-time, Part-time, Contract |
| region | Northeast, South, Midwest, West |
| has_dependents | Yes, No |
| salary | Float, can be null |
| tenure_years | Float, can be null |

---

## Running Tests

```bash
pytest tests/ -v
```

All tests should pass. Tests cover data generation, preprocessing (no NaN after imputation), model fit/predict, probability bounds, and AUC > 0.5.
