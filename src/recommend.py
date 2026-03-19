"""
Action Recommendation Module
Provides business actions based on churn prediction and contributing factors.
Adapted for Kaggle Telco Customer Churn dataset.
"""


# Recommendation rules based on churn probability levels
CHURN_RECOMMENDATIONS = {
    'HIGH': {
        'threshold': 0.7,
        'urgency': 'URGENT',
        'base_action': 'Immediate retention intervention required',
        'actions': [
            'Offer personalized discount (20-30% off monthly charges)',
            'Assign dedicated account manager for personalized outreach',
            'Propose contract upgrade with loyalty discount',
            'Offer free add-on services (security, backup, tech support)'
        ]
    },
    'MODERATE': {
        'threshold': 0.4,
        'urgency': 'MODERATE',
        'base_action': 'Proactive engagement recommended',
        'actions': [
            'Send re-engagement notification with service highlights',
            'Offer limited-time bundle discount (10-15% off)',
            'Highlight value of add-on services via email',
            'Invite to customer loyalty rewards program'
        ]
    },
    'LOW': {
        'threshold': 0.0,
        'urgency': 'LOW',
        'base_action': 'Standard customer maintenance',
        'actions': [
            'Continue regular engagement',
            'Include in loyalty rewards program',
            'Send monthly service newsletter',
            'No immediate action required'
        ]
    }
}

# Factor-specific recommendations for Telco features
# Actions and reasons are now FUNCTIONS that take the value and return
# personalized text — so recommendations change with the customer's data.
FACTOR_RECOMMENDATIONS = {
    'Contract': {
        'condition': lambda x: x == 'Month-to-month',
        'action': lambda v, **kw: f"Offer discounted annual or two-year contract (current: {v}) — lock in loyalty with a {kw.get('discount', 15)}% migration discount",
        'reason': lambda v, **kw: f"Customer is on a {v} contract — the highest churn risk segment"
    },
    'InternetService': {
        'condition': lambda x: x == 'Fiber optic',
        'action': lambda v, **kw: f"Review fiber optic service quality for this customer and offer a bundle discount (monthly: ${kw.get('charges', '?')})",
        'reason': lambda v, **kw: f"Fiber optic customers churn at a higher rate than DSL users"
    },
    'OnlineSecurity': {
        'condition': lambda x: x == 'No',
        'action': lambda v, **kw: f"Offer free {kw.get('trial_months', 3)}-month trial of online security — customers who add it are significantly more likely to stay",
        'reason': lambda v, **kw: f"This customer has no online security — adding it can improve retention"
    },
    'TechSupport': {
        'condition': lambda x: x == 'No',
        'action': lambda v, **kw: f"Offer complimentary tech support for {kw.get('trial_months', 3)} months — this customer currently has none",
        'reason': lambda v, **kw: f"No tech support — customers without it churn at a higher rate"
    },
    'OnlineBackup': {
        'condition': lambda x: x == 'No',
        'action': lambda v, **kw: "Offer free cloud backup service trial — more services means more reasons to stay",
        'reason': lambda v, **kw: f"Customer has no online backup — adding services increases stickiness"
    },
    'DeviceProtection': {
        'condition': lambda x: x == 'No',
        'action': lambda v, **kw: "Offer device protection plan at a discounted rate to add value",
        'reason': lambda v, **kw: "No device protection — bundling more services reduces churn"
    },
    'PaymentMethod': {
        'condition': lambda x: x == 'Electronic check',
        'action': lambda v, **kw: f"Encourage switching from {v} to automatic bank transfer or credit card — offer a one-time incentive",
        'reason': lambda v, **kw: f"This customer pays via {v} — the payment method with the highest churn rate"
    },
    'PaperlessBilling': {
        'condition': lambda x: x == 'Yes',
        'action': lambda v, **kw: "Send clearer billing summaries and usage reports — paperless customers may feel less connected",
        'reason': lambda v, **kw: "Customer uses paperless billing — may feel less engaged with the service"
    },
    'tenure': {
        'condition': lambda x: isinstance(x, (int, float)) and x < 12,
        'action': lambda v, **kw: f"Urgent onboarding follow-up — this customer has only been with us {int(v)} month{'s' if int(v) != 1 else ''} (critical retention window)",
        'reason': lambda v, **kw: f"Tenure is only {int(v)} month{'s' if int(v) != 1 else ''} — new customers in the first year are at highest risk"
    },
    'MonthlyCharges': {
        'condition': lambda x: isinstance(x, (int, float)) and x > 70,
        'action': lambda v, **kw: f"Review pricing for this customer (${v:.2f}/month) — consider a loyalty discount of ${v*0.15:.2f}/month (15% off)",
        'reason': lambda v, **kw: f"Monthly charges are ${v:.2f} — above the $70 threshold where churn risk increases"
    },
    'Partner': {
        'condition': lambda x: x == 'No',
        'action': lambda v, **kw: "Offer family/partner bundle plan — household accounts have lower churn",
        'reason': lambda v, **kw: "Customer has no partner on the account — single-account users churn more"
    },
    'Dependents': {
        'condition': lambda x: x == 'No',
        'action': lambda v, **kw: "Suggest family plans or shared accounts for better value",
        'reason': lambda v, **kw: "Customer has no dependents — family accounts are stickier"
    }
}


def get_risk_level(churn_probability):
    """Determine risk level based on churn probability."""
    if churn_probability >= 0.7:
        return 'HIGH'
    elif churn_probability >= 0.4:
        return 'MODERATE'
    else:
        return 'LOW'


def get_base_recommendation(churn_probability):
    """Get the base recommendation based on churn probability."""
    risk_level = get_risk_level(churn_probability)
    rec = CHURN_RECOMMENDATIONS[risk_level]
    
    return {
        'risk_level': risk_level,
        'urgency': rec['urgency'],
        'primary_action': rec['base_action'],
        'suggested_actions': rec['actions'].copy(),
        'churn_probability': churn_probability
    }


def get_factor_specific_recommendations(customer_data, top_factors=None):
    """
    Get recommendations based on specific customer factors.
    
    Actions and reasons are generated dynamically using the customer's
    actual values and SHAP top-factor information — so the output
    changes whenever the input data changes.
    
    Args:
        customer_data: Dictionary of customer features
        top_factors: List of top SHAP factors (optional)
        
    Returns:
        List of factor-specific recommendations
    """
    recommendations = []
    
    # Build a lookup of SHAP impacts for personalization
    shap_lookup = {}
    if top_factors:
        for f in top_factors:
            shap_lookup[f.get('feature', '')] = {
                'shap_value': f.get('shap_value', 0),
                'abs_shap': f.get('abs_shap_value', abs(f.get('shap_value', 0))),
                'display_name': f.get('display_name', f.get('feature', '')),
                'impact_direction': f.get('impact_direction', '+'),
            }
    
    # Extra context passed to action/reason functions
    extra = {
        'charges': customer_data.get('MonthlyCharges', '?'),
        'tenure': customer_data.get('tenure', '?'),
    }
    
    for feature, rec_info in FACTOR_RECOMMENDATIONS.items():
        if feature in customer_data:
            value = customer_data[feature]
            try:
                triggered = rec_info['condition'](value)
            except (TypeError, ValueError):
                triggered = False
            
            if triggered:
                # Call action/reason as functions for dynamic text
                action_fn = rec_info['action']
                reason_fn = rec_info['reason']
                action_text = action_fn(value, **extra) if callable(action_fn) else action_fn
                reason_text = reason_fn(value, **extra) if callable(reason_fn) else reason_fn
                
                is_top = feature in shap_lookup
                shap_info = shap_lookup.get(feature, {})
                shap_pct = abs(shap_info.get('shap_value', 0)) * 100
                
                # Add SHAP impact to reason if available
                if is_top and shap_pct > 0:
                    reason_text += f" (SHAP impact: {shap_pct:.1f}%)"
                
                recommendations.append({
                    'feature': feature,
                    'value': value,
                    'reason': reason_text,
                    'action': action_text,
                    'is_top_factor': is_top,
                    'shap_impact': shap_pct
                })
    
    # Sort: top factors first (by SHAP impact desc), then others
    def _sort_key(x):
        return (not x['is_top_factor'], -x.get('shap_impact', 0))
    recommendations.sort(key=_sort_key)
    
    return recommendations


def generate_full_recommendation(churn_probability, customer_data, top_factors=None):
    """
    Generate a complete recommendation report.
    
    Args:
        churn_probability: Predicted churn probability
        customer_data: Dictionary of customer features
        top_factors: List of top SHAP factors from explainability module
        
    Returns:
        Complete recommendation dictionary
    """
    # Get base recommendation
    base_rec = get_base_recommendation(churn_probability)
    
    # Get factor-specific recommendations
    factor_recs = get_factor_specific_recommendations(customer_data, top_factors)
    
    # Prioritize actions based on top factors
    prioritized_actions = []
    for factor_rec in factor_recs:
        if factor_rec['is_top_factor']:
            prioritized_actions.insert(0, factor_rec['action'])
        else:
            prioritized_actions.append(factor_rec['action'])
    
    # Combine and deduplicate actions
    all_actions = prioritized_actions + base_rec['suggested_actions']
    unique_actions = list(dict.fromkeys(all_actions))  # Remove duplicates while preserving order
    
    # Generate summary message
    risk_level = base_rec['risk_level']
    if risk_level == 'HIGH':
        summary = f"ALERT: This customer has a {churn_probability*100:.0f}% probability of churning. Immediate action required!"
    elif risk_level == 'MODERATE':
        summary = f"CAUTION: This customer has a {churn_probability*100:.0f}% probability of churning. Proactive engagement recommended."
    else:
        summary = f"OK: This customer has a low churn risk ({churn_probability*100:.0f}%). Continue standard engagement."
    
    return {
        'summary': summary,
        'risk_level': risk_level,
        'urgency': base_rec['urgency'],
        'churn_probability': churn_probability,
        'primary_action': base_rec['primary_action'],
        'recommended_actions': unique_actions[:5],  # Top 5 actions
        'factor_insights': factor_recs,
        'all_suggested_actions': unique_actions
    }


def format_recommendation_report(recommendation):
    """Format recommendation as a readable text report."""
    lines = []
    lines.append("=" * 60)
    lines.append("ACTION RECOMMENDATION REPORT")
    lines.append("=" * 60)
    
    lines.append(f"\n{recommendation['summary']}")
    lines.append(f"\nRisk Level: {recommendation['risk_level']}")
    lines.append(f"Urgency: {recommendation['urgency']}")
    lines.append(f"Churn Probability: {recommendation['churn_probability']*100:.1f}%")
    
    lines.append(f"\n{'-' * 60}")
    lines.append("PRIMARY ACTION:")
    lines.append(f"  >> {recommendation['primary_action']}")
    
    lines.append(f"\n{'-' * 60}")
    lines.append("RECOMMENDED ACTIONS:")
    for i, action in enumerate(recommendation['recommended_actions'], 1):
        lines.append(f"  {i}. {action}")
    
    if recommendation['factor_insights']:
        lines.append(f"\n{'-' * 60}")
        lines.append("FACTOR-SPECIFIC INSIGHTS:")
        for insight in recommendation['factor_insights']:
            marker = "[TOP FACTOR]" if insight['is_top_factor'] else ""
            shap_str = f" (impact: {insight['shap_impact']:.1f}%)" if insight.get('shap_impact', 0) > 0 else ""
            lines.append(f"\n  {insight['feature'].upper()} {marker}{shap_str}")
            lines.append(f"    Reason: {insight['reason']}")
            lines.append(f"    Action: {insight['action']}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def main():
    """Test recommendation system."""
    print("=" * 50)
    print("ACTION RECOMMENDATION TEST")
    print("Telco Customer Churn Dataset")
    print("=" * 50)
    
    # Simulate a high-risk customer (month-to-month, fiber optic, no add-ons)
    high_risk_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.35,
        'TotalCharges': 140.70
    }
    
    # Simulate top factors from SHAP
    top_factors = [
        {'feature': 'Contract', 'shap_value': 0.25},
        {'feature': 'tenure', 'shap_value': 0.18},
        {'feature': 'InternetService', 'shap_value': 0.12}
    ]
    
    print("\n[TEST 1] High-Risk Customer (78% churn probability)")
    print("-" * 50)
    
    rec = generate_full_recommendation(0.78, high_risk_customer, top_factors)
    print(format_recommendation_report(rec))
    
    # Simulate a low-risk customer (two-year contract, DSL, with add-ons)
    low_risk_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 60,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 85.50,
        'TotalCharges': 5130.00
    }
    
    print("\n[TEST 2] Low-Risk Customer (12% churn probability)")
    print("-" * 50)
    
    rec2 = generate_full_recommendation(0.12, low_risk_customer, [])
    print(format_recommendation_report(rec2))
    
    print("\n[OK] Recommendation test complete!")


if __name__ == "__main__":
    main()