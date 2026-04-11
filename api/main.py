from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List
import copy

# Import your modules
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
        # 🟢 STEP 1: Define input_data immediately
        # We use .copy() for the model and deepcopy for recommendations
        input_data = customer.data.copy()
        raw_data_for_recom = copy.deepcopy(customer.data)

        # 🔵 STEP 2: Get Prediction
        # (Note: input_data might be modified/encoded here)
        prediction = predict(input_data)
        prob_value = float(prediction["probability"])

        # 🟡 STEP 3: Get Explanation with Percentage Calculation
        factors = []
        try:
            processed = preprocess_input(input_data, encoders, feature_names)
            exp_res = explain_prediction(processed, model, feature_names)
            
            raw_factors = exp_res.get("top_factors", [])
            
            # Calculate total impact for percentage normalization
            total_impact = sum(abs(f.get("impact", 0)) for f in raw_factors)
            
            for f in raw_factors:
                val = f.get("impact", 0)
                # Normalize to % (fallback to 1 to avoid division by zero)
                pct = (abs(val) / (total_impact if total_impact > 0 else 1)) * 100
                
                factors.append({
                    "feature": f.get("feature_name", f.get("feature")),
                    "impact_pct": round(pct, 1),
                    "direction": f.get("direction")
                })
        except Exception as e:
            print(f"SHAP Error: {e}")

        # 🟠 STEP 4: Get Recommendations
        # We use raw_data_for_recom so the strings like "Basic" are still there
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
        # 🟢 PRE-ALLOCATE: Processing everything in one loop without SHAP
        results = []
        
        # In a real production environment, you'd use model.predict(df) 
        # for vectorization, but for now, we'll keep the loop fast.
        for customer_item in batch.data:
            # Skip copies/deepcopies here to save memory/time
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