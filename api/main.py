from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List
import copy
import sys

# =========================================================
# EMERGENCY MONKEY PATCH FOR SKLEARN 1.8.0 / IMBLEARN
# This must run BEFORE any other internal imports
# =========================================================
import sklearn.utils.validation

def _is_pandas_df(obj):
    try:
        import pandas as pd
        return isinstance(obj, pd.DataFrame)
    except ImportError:
        return False

# Inject the missing function that imblearn is looking for
sklearn.utils.validation._is_pandas_df = _is_pandas_df
# =========================================================

# Now safe to import your modules
from src.predict import predict, load_model_and_encoders, preprocess_input
from src.recommend import generate_recommendations
from src.explain import explain_prediction

app = FastAPI(title="OTT Churn Prediction System")

# Load Model once at startup
model, encoders, feature_names = load_model_and_encoders()

class CustomerData(BaseModel):
    data: Dict[str, Any]

@app.post("/full")
def full_pipeline(customer: CustomerData):
    """
    Combines Prediction, SHAP Explanation (%), and Recommendations.
    """
    try:
        input_data = customer.data.copy()
        raw_data_for_recom = copy.deepcopy(customer.data)

        prediction = predict(input_data)
        prob_value = float(prediction["probability"])

        factors = []
        try:
            processed = preprocess_input(input_data, encoders, feature_names)
            exp_res = explain_prediction(processed, model, feature_names)
            
            raw_factors = exp_res.get("top_factors", [])
            total_impact = sum(abs(f.get("impact", 0)) for f in raw_factors)
            
            for f in raw_factors:
                val = f.get("impact", 0)
                pct = (abs(val) / (total_impact if total_impact > 0 else 1)) * 100
                
                factors.append({
                    "feature": f.get("feature_name", f.get("feature")),
                    "impact_pct": round(pct, 1),
                    "direction": f.get("direction")
                })
        except Exception as e:
            print(f"SHAP Error: {e}")

        recommendation = generate_recommendations(
            prob_value,
            raw_data_for_recom,
            top_factors=factors
        )

        return {
            "prediction": prediction,
            "explanation": factors,
            "recommendation": recommendation
        }

    except Exception as e:
        print(f"Pipeline Error: {e}")
        return {"error": str(e)}

class BatchCustomerData(BaseModel):
    data: List[Dict[str, Any]]

@app.post("/predict_batch")
def predict_batch(batch: BatchCustomerData):
    try:
        results = []
        for customer_item in batch.data:
            prediction = predict(customer_item) 
            
            results.append({
                "Customer_ID": customer_item.get("customer_id", "N/A"),
                "Churn_Probability": round(float(prediction["probability"]), 3),
                "Risk_Level": prediction["risk_level"],
                "Priority": "🔴 High" if prediction["risk_level"] == "HIGH" else "🟢 Low"
            })
            
        return {"status": "success", "predictions": results}
    
    except Exception as e:
        print(f"BATCH API ERROR: {e}")
        return {"status": "error", "message": str(e)}