"""
FastAPI service for diabetes progression prediction.
Provides /predict endpoint for triage nurse dashboard.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Diabetes Progression Prediction Service",
    description="ML service for predicting short-term diabetes progression risk",
    version="0.1.0"
)

# Global model storage
MODEL = None
SCALER = None


class PatientFeatures(BaseModel):
    """
    Patient features matching scikit-learn diabetes dataset structure.
    In production, these would be real EHR features (vitals, labs, lifestyle).
    """
    age: float = Field(..., description="Age (standardized)")
    sex: float = Field(..., description="Sex (standardized)")
    bmi: float = Field(..., description="Body mass index (standardized)")
    bp: float = Field(..., description="Average blood pressure (standardized)")
    s1: float = Field(..., description="TC, total serum cholesterol (standardized)")
    s2: float = Field(..., description="LDL, low-density lipoproteins (standardized)")
    s3: float = Field(..., description="HDL, high-density lipoproteins (standardized)")
    s4: float = Field(..., description="TCH, total cholesterol / HDL (standardized)")
    s5: float = Field(..., description="LTG, log of serum triglycerides (standardized)")
    s6: float = Field(..., description="GLU, blood sugar level (standardized)")
    
    @validator('*', pre=True)
    def check_numeric(cls, v):
        """Validate all fields are numeric."""
        if not isinstance(v, (int, float)):
            raise ValueError(f"Must be numeric, got {type(v)}")
        if np.isnan(v) or np.isinf(v):
            raise ValueError("Cannot be NaN or Inf")
        return float(v)


class PredictionRequest(BaseModel):
    """Request body for batch predictions."""
    patients: List[PatientFeatures]


class PredictionResponse(BaseModel):
    """Response with progression risk scores."""
    patient_id: int
    progression_score: float
    risk_level: str
    high_risk: bool


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    model_version: str


def load_model(model_path: str = "models/model_v0.1.pkl"):
    """Load trained model and scaler."""
    global MODEL, SCALER
    
    try:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_file, "rb") as f:
            artifacts = pickle.load(f)
        
        SCALER = artifacts["scaler"]
        MODEL = artifacts["model"]
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Model type: {type(MODEL).__name__}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def classify_risk(score: float, threshold: float = 150.0) -> tuple:
    """
    Classify progression score into risk categories.
    
    Args:
        score: Predicted progression score
        threshold: High-risk threshold (default: 75th percentile ~150)
    
    Returns:
        (risk_level, is_high_risk)
    """
    if score >= threshold:
        return "HIGH", True
    elif score >= threshold * 0.75:
        return "MEDIUM", False
    else:
        return "LOW", False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting Diabetes Progression Prediction Service...")
    load_model()
    logger.info("Service ready!")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Diabetes Progression Prediction",
        "status": "healthy",
        "model_loaded": MODEL is not None
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model": type(MODEL).__name__,
        "scaler": type(SCALER).__name__
    }


@app.post("/predict", response_model=BatchPredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict diabetes progression for one or more patients.
    
    Returns progression scores sorted by risk (descending) for triage dashboard.
    """
    if MODEL is None or SCALER is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        # Convert patient features to array
        feature_names = [
            "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"
        ]
        
        X = np.array([
            [getattr(patient, feat) for feat in feature_names]
            for patient in request.patients
        ])
        
        # Scale and predict
        X_scaled = SCALER.transform(X)
        predictions = MODEL.predict(X_scaled)
        
        # Create response with risk classification
        results = []
        for idx, score in enumerate(predictions):
            risk_level, is_high_risk = classify_risk(float(score))
            
            results.append(PredictionResponse(
                patient_id=idx,
                progression_score=float(score),
                risk_level=risk_level,
                high_risk=is_high_risk
            ))
        
        # Sort by progression score (descending) for triage dashboard
        results.sort(key=lambda x: x.progression_score, reverse=True)
        
        return BatchPredictionResponse(
            predictions=results,
            model_version="v0.1"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input features: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/single")
async def predict_single(patient: PatientFeatures):
    """Convenience endpoint for single patient prediction."""
    request = PredictionRequest(patients=[patient])
    response = await predict(request)
    return response.predictions[0]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
