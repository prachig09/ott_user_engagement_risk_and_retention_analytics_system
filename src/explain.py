
#Explainability Module (OTT Version)
#Provides SHAP-based explanations for churn predictions.



import os
import joblib
import numpy as np
import shap




# ✅ YOUR DATASET FEATURES
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
   """Return SHAP explainer for tree-based models."""
   return shap.Explainer(model)




def explain_prediction(features, model, feature_names):
   """
   Generate SHAP explanation for a prediction.
   """


   explainer = get_explainer(model)
   shap_values = explainer(features)


   values = shap_values.values[0]


   explanations = []


   for i, feature in enumerate(feature_names):
       impact = values[i]


       explanations.append({
           "feature": feature,
           "feature_name": FEATURE_DESCRIPTIONS.get(feature, feature),
           "impact": float(impact),
           "direction": "increase" if impact > 0 else "decrease"
       })


   # Sort by importance
   explanations = sorted(explanations, key=lambda x: abs(x['impact']), reverse=True)


   return {
       "top_factors": explanations[:3],
       "all_factors": explanations
   }




# ✅ TEST
def main():
   from predict import predict_churn, preprocess_customer_data, load_model_and_encoders


   model, encoders, feature_names = load_model_and_encoders()


   sample_customer = {
       "age": 25,
       "gender": "Male",
       "subscription_type": "Basic",
       "monthly_charges": 199,
       "tenure_in_months": 2,
       "login_frequency": 1,
       "last_login_days": 10,
       "watch_time": 2,
       "payment_failures": 1,
       "customer_support_calls": 3
   }


   features = preprocess_customer_data(sample_customer, encoders, feature_names)


   prediction = predict_churn(sample_customer, model, encoders, feature_names)


   explanation = explain_prediction(features, model, feature_names)


   print("\n🎯 Prediction:", prediction["prediction_label"])
   print("📊 Probability:", prediction["churn_probability"])


   print("\n🔍 Top Factors:")
   for f in explanation["top_factors"]:
       print(f" - {f['feature_name']} ({f['direction']}, impact={f['impact']:.4f})")




if __name__ == "__main__":
   main()