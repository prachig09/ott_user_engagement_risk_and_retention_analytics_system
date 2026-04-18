import os
import copy
import uvicorn
import gradio as gr
import gradio.routes
from gradio_client import utils as client_utils
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List

# --- 0. EMERGENCY GRADIO API PATCH ---
# This fixes the "TypeError: argument of type 'bool' is not iterable" 
# by preventing Gradio from trying to parse the FastAPI schema.
def dummy_api_info(*args, **kwargs):
    return {"components": [], "endpoints": {}}

gradio.routes.api_info = dummy_api_info
client_utils.json_schema_to_python_type = lambda x: "Any"

# --- 1. INTERNAL IMPORTS ---
from src.predict import batch_predict, predict, load_model_and_encoders, preprocess_input
from src.recommend import generate_recommendations
from src.explain import explain_prediction
from main import demo, theme, CSS 

app = FastAPI(title="OTT Churn Prediction System")

# Load model/encoders globally
model, encoders, feature_names = load_model_and_encoders()

class CustomerData(BaseModel):
    data: Dict[str, Any]

class BatchCustomerData(BaseModel):
    data: List[Dict[str, Any]]

# --- 2. API ENDPOINTS ---
@app.post("/full")
def full_pipeline(customer: CustomerData):
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
            
        recommendation = generate_recommendations(prob_value, raw_data_for_recom, top_factors=factors)
        return {"prediction": prediction, "explanation": factors, "recommendation": recommendation}
    except Exception as e: 
        return {"error": str(e)}

@app.post("/predict_batch")
def predict_batch(batch: BatchCustomerData):
    return batch_predict(batch.data)

# --- 3. GRADIO MOUNTING ---
head_injection = f"<style>{CSS}</style>"

# Force show_api to False on the blocks object
demo.show_api = False

app = gr.mount_gradio_app(
    app, 
    demo, 
    path="/",
    app_kwargs={
        "theme": theme,
        "css": CSS,
        "title": "OTT Retention System",
        "head": head_injection 
    }
)

if __name__ == "__main__":
    # Use log_level="info" to see connections, port 7860 for Docker compatibility
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")