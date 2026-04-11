import os
import joblib
import pandas as pd


def load_model_and_encoders():
    """Load trained model and preprocessing objects."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model = joblib.load(os.path.join(base_dir, "model", "churn_model.pkl"))
    encoders = joblib.load(os.path.join(base_dir, "model", "encoder.pkl"))
    feature_names = joblib.load(os.path.join(base_dir, "model", "feature_names.pkl"))

    return model, encoders, feature_names


def preprocess_input(data, encoders, feature_names):
    """Preprocess input data for prediction."""
    
    # 1. Force keys to lowercase to match CSV: 'Gender' -> 'gender'
    clean_data = {k.lower().replace(" ", "_"): v for k, v in data.items()}
    
    # 2. Force categorical values to lowercase: 'Basic' -> 'basic'
    for k, v in clean_data.items():
        if isinstance(v, str):
            clean_data[k] = v.lower()

    df = pd.DataFrame([clean_data])

    # 3. One-hot encode
    df = pd.get_dummies(df)

    # 4. Align with training features (This is where the 0s were coming from)
    df = df.reindex(columns=feature_names, fill_value=0)

    # 5. Scale
    scaler = encoders["scaler"]
    df_scaled = scaler.transform(df)

    return df_scaled


def predict(data):
    """Predict churn for a new customer."""

    model, encoders, feature_names = load_model_and_encoders()

    processed = preprocess_input(data, encoders, feature_names)

    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0][1]

    # 🔹 Risk logic
    if prob >= 0.7:
        risk = "HIGH"
    elif prob >= 0.4:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return {
        "prediction": int(pred),
        "label": "Churn" if pred == 1 else "Stay",
        "probability": round(float(prob), 3),
        "risk_level": risk
    }


# 🔥 TEST
if __name__ == "__main__":

    sample_user = {
        "age": 28,
        "gender": "Female",
        "subscription_type": "Basic",
        "monthly_charges": 199,
        "tenure_in_months": 3,
        "login_frequency": 4,
        "last_login_days": 20,
        "watch_time": 5,
        "payment_failures": 1,
        "customer_support_calls": 2
    }

    result = predict(sample_user)

    print("\n🎯 Prediction Result")
    print("=" * 40)
    print(f"Prediction: {result['label']}")
    print(f"Probability: {result['probability']}")
    print(f"Risk Level: {result['risk_level']}")