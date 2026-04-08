"""
FastAPI for OTT Churn Prediction System

Endpoints:
- GET  /            → Health check
- POST /predict     → Prediction only
- POST /recommend   → Prediction + Recommendations
- POST /full        → Combined output
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

# Import your modules
from src.predict import predict, load_model_and_encoders
from src.recommend import generate_recommendations

# ─────────────────────────────────────────────
# Initialize App
# ─────────────────────────────────────────────
app = FastAPI(
    title="OTT Churn Prediction API",
    description="Predict churn and generate retention strategies",
    version="1.0"
)

# ─────────────────────────────────────────────
# Load Model ONCE (IMPORTANT 🚀)
# ─────────────────────────────────────────────
model, encoders, feature_names = load_model_and_encoders()


# ─────────────────────────────────────────────
# Request Schema
# ─────────────────────────────────────────────
class CustomerData(BaseModel):
    data: Dict[str, Any]


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status": "OK",
        "message": "OTT Churn API is running 🚀"
    }


# ─────────────────────────────────────────────
# Prediction Endpoint
# ─────────────────────────────────────────────
@app.post("/predict")
def predict_api(customer: CustomerData):
    """
    Returns churn prediction only
    """
    result = predict(customer.data)
    return result


# ─────────────────────────────────────────────
# Recommendation Endpoint
# ─────────────────────────────────────────────
@app.post("/recommend")
def recommend_api(customer: CustomerData):
    """
    Returns prediction + recommendations
    """
    prediction = predict(customer.data)

    recommendation = generate_recommendations(
        prediction["probability"],
        customer.data
    )

    return {
        "prediction": prediction,
        "recommendation": recommendation
    }


# ─────────────────────────────────────────────
# Full Pipeline Endpoint
# ─────────────────────────────────────────────
@app.post("/full")
def full_pipeline(customer: CustomerData):
    """
    Full pipeline:
    - Prediction
    - Recommendations
    """

    prediction = predict(customer.data)

    recommendation = generate_recommendations(
        prediction["probability"],
        customer.data
    )

    return {
        "prediction": prediction,
        "recommendation": recommendation
    }