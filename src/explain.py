import os
import joblib
import numpy as np
import shap

# ✅ FRIENDLY DESCRIPTIONS
FEATURE_DESCRIPTIONS = {
   'age': 'Customer Age',
   'gender': 'Gender',
   'subscription_type': 'Subscription Type',
   'monthly_charges': 'Monthly Charges',
   'tenure_in_months': 'Tenure (Months)',
   'login_frequency': 'Login Frequency',
   'last_login_days': 'Days Since Last Login',
   'watch_time': 'Watch Time',
   'payment_failures': 'Payment Failures',
   'customer_support_calls': 'Support Calls'
}

def load_model():
   script_dir = os.path.dirname(os.path.abspath(__file__))
   project_dir = os.path.dirname(script_dir)
   model_dir = os.path.join(project_dir, 'model')

   model = joblib.load(os.path.join(model_dir, 'churn_model.pkl'))
   feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))

   return model, feature_names

def get_explainer(model):
    """
    FIXED: Returns SHAP explainer for the classifier inside the pipeline.
    Your error occurred because SHAP tried to analyze the 'Pipeline' object 
    instead of the actual 'GradientBoostingClassifier' inside it.
    """
    # Check if the model is a Scikit-Learn Pipeline
    if hasattr(model, 'named_steps'):
        # Extract the classifier step (named 'model' in your error log)
        actual_classifier = model.named_steps['model']
        return shap.Explainer(actual_classifier)
    
    # Fallback for standard models
    return shap.Explainer(model)

def explain_prediction(features, model, feature_names):
    """
    Generate SHAP explanation for a prediction.
    """
    explainer = get_explainer(model)
    shap_values = explainer(features)
    
    # Extract values for the first (and only) row
    values = shap_values.values[0]

    explanations = []

    for i, col_name in enumerate(feature_names):
        impact = values[i]
        
        # 🔹 SMART MAPPING
        # Logic to map encoded names (subscription_type_premium) back to (subscription_type)
        base_feature = col_name
        for original in FEATURE_DESCRIPTIONS.keys():
            if col_name.startswith(original):
                base_feature = original
                break

        explanations.append({
            "feature": base_feature,
            "feature_name": FEATURE_DESCRIPTIONS.get(base_feature, base_feature),
            "impact": float(impact),
            "direction": "increase" if impact > 0 else "decrease"
        })

    # 🔹 SORT BY IMPORTANCE
    # We use abs(impact) because a strong negative impact is just as important as a positive one
    explanations = sorted(explanations, key=lambda x: abs(x['impact']), reverse=True)

    return {
        "top_factors": explanations[:3], # Return top 3 for the UI
        "all_factors": explanations
    }

# ✅ TEST SCRIPT
if __name__ == "__main__":
    from predict import preprocess_input, load_model_and_encoders
    
    model, encoders, feature_names = load_model_and_encoders()

    sample_customer = {
        "age": 22,
        "gender": "female",
        "subscription_type": "basic",
        "monthly_charges": 19.99,
        "tenure_in_months": 2,
        "login_frequency": 2,
        "last_login_days": 25,
        "watch_time": 15,  # 0.5hr/wk * 30 scale
        "payment_failures": 3,
        "customer_support_calls": 4
    }

    # Preprocess
    features = preprocess_input(sample_customer, encoders, feature_names)

    # Explain
    explanation = explain_prediction(features, model, feature_names)

    print("\n🔍 SHAP Top Factors Analysis:")
    print("=" * 40)
    for f in explanation["top_factors"]:
        status = "🔴 Increases Risk" if f['direction'] == "increase" else "🟢 Decreases Risk"
        print(f" - {f['feature_name']}: {status} (Impact: {f['impact']:.4f})")