"""
src/api.py
----------
FastAPI REST API for serving insurance enrollment predictions.

Endpoints:
  POST /predict       - single employee prediction
  POST /predict/batch - batch prediction (list of employees)
  GET  /health        - health check
  GET  /model/info    - loaded model metadata

Run with:
  uvicorn src.api:app --reload --port 8000

Test with:
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"age":35,"gender":"Female","marital_status":"Married","salary":75000,
         "employment_type":"Full-time","region":"Northeast","has_dependents":"Yes","tenure_years":5}'
"""

import os
import json
import pickle
from typing import List, Optional
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# -- Load model at startup -----------------------------------------------------
MODEL_PATH = "models/best_model.pkl"
_pipeline  = None

def load_model():
    global _pipeline
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run: python train.py")
    with open(MODEL_PATH, "rb") as f:
        _pipeline = pickle.load(f)
    print(f"[api] Model loaded from {MODEL_PATH}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler — loads model on startup."""
    load_model()
    yield


# -- App setup -----------------------------------------------------------------
app = FastAPI(
    title="Insurance Enrollment Predictor",
    description="Predicts likelihood of employee voluntary insurance enrollment.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# REQUEST / RESPONSE SCHEMAS
# ==============================================================================

class EmployeeInput(BaseModel):
    age:             int   = Field(..., ge=18, le=75, description="Employee age")
    gender:          str   = Field(..., description="Male | Female | Other")
    marital_status:  str   = Field(..., description="Single | Married | Divorced | Widowed")
    salary:          Optional[float] = Field(None, ge=0, description="Annual salary (can be null)")
    employment_type: str   = Field(..., description="Full-time | Part-time | Contract")
    region:          str   = Field(..., description="Northeast | South | Midwest | West")
    has_dependents:  str   = Field(..., description="Yes | No")
    tenure_years:    Optional[float] = Field(None, ge=0, description="Years at company (can be null)")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        allowed = {"Male", "Female", "Other"}
        if v not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v

    @field_validator("marital_status")
    @classmethod
    def validate_marital(cls, v):
        allowed = {"Single", "Married", "Divorced", "Widowed"}
        if v not in allowed:
            raise ValueError(f"marital_status must be one of {allowed}")
        return v

    @field_validator("employment_type")
    @classmethod
    def validate_emp_type(cls, v):
        allowed = {"Full-time", "Part-time", "Contract"}
        if v not in allowed:
            raise ValueError(f"employment_type must be one of {allowed}")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v):
        allowed = {"Northeast", "South", "Midwest", "West"}
        if v not in allowed:
            raise ValueError(f"region must be one of {allowed}")
        return v

    @field_validator("has_dependents")
    @classmethod
    def validate_dependents(cls, v):
        allowed = {"Yes", "No"}
        if v not in allowed:
            raise ValueError(f"has_dependents must be one of {allowed}")
        return v


class PredictionResponse(BaseModel):
    enrolled_probability: float = Field(..., description="Probability of enrollment (0–1)")
    prediction:           int   = Field(..., description="1=likely to enroll, 0=not likely")
    confidence:           str   = Field(..., description="High / Medium / Low based on probability")


def _to_dataframe(employee: EmployeeInput) -> pd.DataFrame:
    """Convert a single EmployeeInput to a DataFrame row the pipeline can process."""
    return pd.DataFrame([{
        "age":             employee.age,
        "gender":          employee.gender,
        "marital_status":  employee.marital_status,
        "salary":          employee.salary,
        "employment_type": employee.employment_type,
        "region":          employee.region,
        "has_dependents":  employee.has_dependents,
        "tenure_years":    employee.tenure_years,
    }])


def _confidence_label(prob: float) -> str:
    if prob >= 0.70 or prob <= 0.30:
        return "High"
    elif prob >= 0.55 or prob <= 0.45:
        return "Medium"
    return "Low"


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/health", tags=["System"])
def health_check():
    """Returns OK if the model is loaded and the API is running."""
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.get("/model/info", tags=["System"])
def model_info():
    """Returns basic info about the loaded model."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    classifier = _pipeline.named_steps.get("classifier")
    return {
        "model_type": type(classifier).__name__,
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(employee: EmployeeInput):
    """
    Predict insurance enrollment likelihood for a single employee.
    Returns probability, binary prediction, and confidence label.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df   = _to_dataframe(employee)
        prob = float(_pipeline.predict_proba(df)[0][1])
        pred = int(prob >= 0.5)
        return PredictionResponse(
            enrolled_probability=round(prob, 4),
            prediction=pred,
            confidence=_confidence_label(prob),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(employees: List[EmployeeInput]):
    """
    Batch predict for a list of employees.
    Returns a list of predictions in the same order as input.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(employees) > 1000:
        raise HTTPException(status_code=400, detail="Max batch size is 1000")

    try:
        rows = [_to_dataframe(e) for e in employees]
        df   = pd.concat(rows, ignore_index=True)
        probs = _pipeline.predict_proba(df)[:, 1]
        return [
            {
                "enrolled_probability": round(float(p), 4),
                "prediction":           int(p >= 0.5),
                "confidence":           _confidence_label(float(p)),
            }
            for p in probs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
