"""
Explainability Module
Uses SHAP / model coefficients to explain individual churn predictions.
Adapted for Kaggle Telco Customer Churn dataset.

Updated to handle:
  - OneHotEncoded features (aggregated back to parent for display)
  - Engineered features (num_services, avg_charge_per_month)
"""

import os
import joblib
import numpy as np
import shap


# ────────────────────────────────────────────────────────────────────
#  Feature descriptions (parent-level, used for display)
# ────────────────────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    'gender': {
        'name': 'Gender',
        'high_impact': 'Gender may influence service preferences'
    },
    'SeniorCitizen': {
        'name': 'Senior Citizen',
        'high_impact': 'Senior citizen status affects churn patterns'
    },
    'Partner': {
        'name': 'Partner',
        'high_impact': 'Having a partner correlates with lower churn'
    },
    'Dependents': {
        'name': 'Dependents',
        'high_impact': 'Customers with dependents tend to stay longer'
    },
    'tenure': {
        'name': 'Tenure',
        'high_impact': 'Longer tenure indicates customer loyalty',
        'unit': 'months'
    },
    'PhoneService': {
        'name': 'Phone Service',
        'high_impact': 'Phone service subscription status'
    },
    'MultipleLines': {
        'name': 'Multiple Lines',
        'high_impact': 'Multiple phone lines indicate deeper engagement'
    },
    'InternetService': {
        'name': 'Internet Service',
        'high_impact': 'Internet service type strongly affects churn'
    },
    'OnlineSecurity': {
        'name': 'Online Security',
        'high_impact': 'Lack of online security increases churn risk'
    },
    'OnlineBackup': {
        'name': 'Online Backup',
        'high_impact': 'Online backup service affects retention'
    },
    'DeviceProtection': {
        'name': 'Device Protection',
        'high_impact': 'Device protection adds customer stickiness'
    },
    'TechSupport': {
        'name': 'Tech Support',
        'high_impact': 'Lack of tech support increases churn risk'
    },
    'StreamingTV': {
        'name': 'Streaming TV',
        'high_impact': 'Streaming TV usage affects engagement'
    },
    'StreamingMovies': {
        'name': 'Streaming Movies',
        'high_impact': 'Streaming movies usage affects engagement'
    },
    'Contract': {
        'name': 'Contract Type',
        'high_impact': 'Month-to-month contracts have highest churn'
    },
    'PaperlessBilling': {
        'name': 'Paperless Billing',
        'high_impact': 'Paperless billing correlates with churn behavior'
    },
    'PaymentMethod': {
        'name': 'Payment Method',
        'high_impact': 'Electronic check users churn more frequently'
    },
    'MonthlyCharges': {
        'name': 'Monthly Charges',
        'high_impact': 'Higher monthly charges may affect retention',
        'unit': '$'
    },
    'TotalCharges': {
        'name': 'Total Charges',
        'high_impact': 'Total charges reflect customer lifetime value',
        'unit': '$'
    },
    'num_services': {
        'name': 'Number of Services',
        'high_impact': 'More subscribed services reduce churn risk'
    },
    'avg_charge_per_month': {
        'name': 'Avg Charge / Month',
        'high_impact': 'Average charge per month of tenure',
        'unit': '$'
    },
    'tenure_group': {
        'name': 'Tenure Group',
        'high_impact': 'Tenure band captures non-linear churn risk'
    }
}


def load_model_and_data():
    """Load the trained model and encoders."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, 'model')

    model = joblib.load(os.path.join(model_dir, 'churn_model.pkl'))
    encoders = joblib.load(os.path.join(model_dir, 'encoder.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))

    return model, encoders, feature_names


def create_explainer(model, X_background=None):
    """Create SHAP explainer (or None for coefficient-based models)."""
    model_type = type(model).__name__
    if model_type in ('RandomForestClassifier', 'GradientBoostingClassifier',
                      'XGBClassifier'):
        return shap.TreeExplainer(model)
    return None


# ────────────────────────────────────────────────────────────────────
#  Core explanation function
# ────────────────────────────────────────────────────────────────────

def explain_prediction(customer_features, model, feature_names,
                       original_values=None, encoders=None):
    """
    Explain a churn prediction using SHAP values or model coefficients.

    For tree models  → real SHAP via TreeExplainer
    For linear models → coefficient × feature contribution

    One-hot dummy features are aggregated back to their parent feature
    so the user sees "Contract Type" instead of "Contract_Month-to-month".

    Args:
        customer_features: Preprocessed (scaled) feature array  (1, n_features)
        model:             Trained model
        feature_names:     List of feature column names
        original_values:   Raw customer dict (for display values)
        encoders:          Encoder dict (for dummy→parent mapping)

    Returns:
        dict with keys: all_features, top_factors, explanations, base_value
    """
    model_type = type(model).__name__

    # ── Raw per-column contributions ────────────────────────────────
    if model_type == 'LogisticRegression':
        coefficients = model.coef_[0]
        shap_values_churn = coefficients * customer_features[0]
        base_value = model.intercept_[0]

    elif model_type in ('RandomForestClassifier', 'GradientBoostingClassifier',
                        'XGBClassifier'):
        explainer = shap.TreeExplainer(model)
        shap_result = explainer(customer_features)
        if len(shap_result.values.shape) == 3:
            shap_values_churn = shap_result.values[0, :, 1]
        else:
            shap_values_churn = shap_result.values[0]
        if hasattr(shap_result, 'base_values'):
            bv = shap_result.base_values
            if isinstance(bv, np.ndarray):
                base_value = float(bv[0][1]) if len(bv.shape) > 1 else float(bv[0])
            else:
                base_value = float(bv)
        else:
            base_value = 0.5
    else:
        shap_values_churn = customer_features[0]
        base_value = 0.5

    # ── Aggregate one-hot dummies back to parent features ───────────
    dummy_to_parent = {}
    if encoders is not None:
        dummy_to_parent = encoders.get('_dummy_to_parent', {})

    # Group contributions by parent feature
    parent_contributions = {}   # parent → sum of contributions
    parent_abs = {}             # parent → sum of |contributions|

    for i, fname in enumerate(feature_names):
        parent = dummy_to_parent.get(fname, fname)  # self if not a dummy
        parent_contributions[parent] = parent_contributions.get(parent, 0.0) + float(shap_values_churn[i])
        parent_abs[parent] = parent_abs.get(parent, 0.0) + abs(float(shap_values_churn[i]))

    # ── Build feature importance list (parent-level) ────────────────
    feature_importance = []
    seen_parents = set()

    for i, fname in enumerate(feature_names):
        parent = dummy_to_parent.get(fname, fname)
        if parent in seen_parents:
            continue
        seen_parents.add(parent)

        feature_info = FEATURE_DESCRIPTIONS.get(parent, {'name': parent})
        shap_val = parent_contributions[parent]
        abs_val  = parent_abs[parent]

        # Get original display value
        if original_values and parent in original_values:
            orig_val = original_values[parent]
        elif original_values and fname in original_values:
            orig_val = original_values[fname]
        else:
            orig_val = customer_features[0][i]

        impact = "increases" if shap_val > 0 else "decreases"
        impact_dir = "+" if shap_val > 0 else "-"

        unit = feature_info.get('unit', '')
        value_str = f"{orig_val} {unit}" if unit else str(orig_val)
        explanation_text = f"{feature_info['name']} ({value_str}) {impact} churn risk"

        feature_importance.append({
            'feature': parent,
            'display_name': feature_info['name'],
            'shap_value': float(shap_val),
            'abs_shap_value': float(abs_val),
            'impact_direction': impact_dir,
            'value': orig_val,
            'explanation': explanation_text
        })

    # Sort by absolute aggregated importance
    feature_importance.sort(key=lambda x: x['abs_shap_value'], reverse=True)

    top_factors = feature_importance[:3]

    explanations = []
    for f in top_factors:
        pct = abs(f['shap_value']) * 100
        sign = '+' if f['impact_direction'] == '+' else ''
        explanations.append(f"{f['explanation']} ({sign}{pct:.1f}%)")

    return {
        'all_features': feature_importance,
        'top_factors': top_factors,
        'explanations': explanations,
        'base_value': float(base_value)
    }


# ────────────────────────────────────────────────────────────────────
#  CLI test
# ────────────────────────────────────────────────────────────────────

def main():
    """Test explainability with sample data."""
    print("=" * 50)
    print("SHAP EXPLAINABILITY TEST")
    print("Telco Customer Churn Dataset")
    print("=" * 50)

    from predict import predict_churn, load_model_and_encoders, preprocess_customer_data

    customer = {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 70.35, 'TotalCharges': 140.70
    }

    model, encoders, feature_names = load_model_and_encoders()

    prediction = predict_churn(customer, model, encoders, feature_names)
    print(f"\n   Churn Probability: {prediction['churn_probability']*100:.1f}%")
    print(f"   Risk Level: {prediction['risk_level']}")

    features = preprocess_customer_data(customer, encoders, feature_names)
    explanation = explain_prediction(features, model, feature_names, customer, encoders)

    print("\n" + "-" * 50)
    print("TOP FACTORS:")
    print("-" * 50)
    for i, f in enumerate(explanation['top_factors'], 1):
        print(f"   {i}. {f['explanation']}")
        print(f"      Impact: {f['impact_direction']}{abs(f['shap_value']):.4f}")

    print("\n" + "-" * 50)
    print("ALL FEATURES:")
    print("-" * 50)
    for f in explanation['all_features']:
        bar_len = int(abs(f['shap_value']) * 30)
        bar = ("+" if f['shap_value'] > 0 else "-") * max(bar_len, 1)
        print(f"   {f['display_name']:25s} {f['impact_direction']}{abs(f['shap_value']):.4f} {bar}")

    print("\n" + "=" * 50)
    print("[OK] Explainability test complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()