"""
Prediction Module
Loads the trained model and makes churn predictions for new customers.
Uses Kaggle Telco Customer Churn dataset features.

Updated to handle:
  - OneHotEncoded multi-class features
  - Engineered features (num_services, avg_charge_per_month)
"""

import os
import joblib
import numpy as np
import pandas as pd


def load_model_and_encoders():
    """Load the trained model and encoders from disk."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, 'model')

    model_path = os.path.join(model_dir, 'churn_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train_model.py first.")

    model = joblib.load(model_path)
    encoders = joblib.load(os.path.join(model_dir, 'encoder.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))

    return model, encoders, feature_names


def preprocess_customer_data(customer_data, encoders=None, feature_names=None):
    """
    Preprocess a single customer's data for prediction.

    Handles:
      - Feature engineering (num_services, avg_charge_per_month)
      - LabelEncoding for binary columns
      - OneHotEncoding for multi-class columns
      - Correct feature ordering & scaling

    Args:
        customer_data: Dictionary with raw customer features
        encoders:      Fitted encoders dict (loaded if None)
        feature_names: Feature name list   (loaded if None)

    Returns:
        Scaled numpy array ready for model.predict()
    """
    if encoders is None or feature_names is None:
        _, encoders, feature_names = load_model_and_encoders()

    data = customer_data.copy()

    # ── Feature engineering ─────────────────────────────────────────
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    data['num_services'] = sum(1 for c in service_cols if data.get(c) == 'Yes')

    tenure = data.get('tenure', 1)
    tenure = max(int(tenure), 1)
    data['avg_charge_per_month'] = float(data.get('TotalCharges', 0)) / tenure

    # Tenure group (must match training bins)
    if tenure <= 12:
        data['tenure_group'] = '0-12'
    elif tenure <= 24:
        data['tenure_group'] = '13-24'
    elif tenure <= 48:
        data['tenure_group'] = '25-48'
    else:
        data['tenure_group'] = '49-72'

    # ── Binary encoding ─────────────────────────────────────────────
    binary_cols = encoders.get('_binary_cols', [])
    for col in binary_cols:
        if col in data and isinstance(data[col], str):
            data[col] = encoders[col].transform([data[col]])[0]

    # ── Build feature vector ────────────────────────────────────────
    multi_cols = encoders.get('_multi_cols', [])
    row = {}

    for fname in feature_names:
        # Check if this is a direct (non-dummy) feature
        if fname in data and fname not in multi_cols:
            # Check it's not a dummy column whose parent is also in data
            parent = encoders.get('_dummy_to_parent', {}).get(fname)
            if parent is None:
                row[fname] = float(data[fname])
                continue

        # Check if this is a one-hot dummy column
        parent = encoders.get('_dummy_to_parent', {}).get(fname)
        if parent is not None and parent in data:
            # e.g. fname = "Contract_Month-to-month", parent = "Contract"
            category = fname[len(parent) + 1:]  # strip "Contract_"
            row[fname] = 1.0 if str(data[parent]) == category else 0.0
        else:
            row[fname] = 0.0  # default for unknown features

    # Create feature array in correct order (as DataFrame to suppress sklearn warning)
    features = pd.DataFrame([[row[fname] for fname in feature_names]], columns=feature_names)

    # Scale
    features_scaled = encoders['scaler'].transform(features)

    return features_scaled


def predict_churn(customer_data, model=None, encoders=None, feature_names=None):
    """
    Predict churn probability for a customer.

    Args:
        customer_data: Dictionary with raw customer features
        model / encoders / feature_names: optional, loaded if not provided

    Returns:
        Dictionary with prediction, probability, risk level
    """
    if model is None or encoders is None or feature_names is None:
        model, encoders, feature_names = load_model_and_encoders()

    features = preprocess_customer_data(customer_data, encoders, feature_names)

    prediction = model.predict(features)[0]

    if hasattr(model, 'predict_proba'):
        churn_probability = model.predict_proba(features)[0][1]
    else:
        churn_probability = float(prediction)

    if churn_probability >= 0.7:
        risk_level = "HIGH"
    elif churn_probability >= 0.4:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        'prediction': int(prediction),
        'prediction_label': 'Churn' if prediction == 1 else 'Stay',
        'churn_probability': float(churn_probability),
        'stay_probability': float(1 - churn_probability),
        'risk_level': risk_level
    }


def main():
    """Test prediction with sample customer data."""
    print("=" * 50)
    print("CUSTOMER CHURN PREDICTION TEST")
    print("Telco Customer Churn Dataset")
    print("=" * 50)

    high_risk_customer = {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 70.35, 'TotalCharges': 140.70
    }

    low_risk_customer = {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes',
        'tenure': 60, 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
        'InternetService': 'DSL', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes', 'TechSupport': 'Yes', 'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes', 'Contract': 'Two year', 'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)', 'MonthlyCharges': 85.50,
        'TotalCharges': 5130.00
    }

    model, encoders, feature_names = load_model_and_encoders()

    for label, customer in [("High-Risk", high_risk_customer),
                            ("Low-Risk", low_risk_customer)]:
        print(f"\n[TEST] {label} Customer:")
        print("-" * 40)
        for k, v in customer.items():
            print(f"   {k}: {v}")
        result = predict_churn(customer, model, encoders, feature_names)
        print(f"\n   Churn Probability: {result['churn_probability']*100:.1f}%")
        print(f"   Prediction: {result['prediction_label']}")
        print(f"   Risk Level: {result['risk_level']}")

    print("\n" + "=" * 50)
    print("[OK] Prediction test complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()