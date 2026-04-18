import gradio as gr
import copy
# No requests needed
from src.predict import predict, load_model_and_encoders, preprocess_input
from src.explain import explain_prediction
from src.recommend import generate_recommendations

# Load tools once for the UI
model, encoders, feature_names = load_model_and_encoders()

def get_prediction(age, gender, sub_type, charges, tenure, login_freq, last_login, watch_time, failures, calls):
   payload_data = {
       "age": int(age),
       "gender": str(gender),
       "subscription_type": str(sub_type),
       "monthly_charges": float(charges),
       "tenure_in_months": int(tenure),
       "login_frequency": int(login_freq),
       "last_login_days": int(last_login),
       "watch_time": float(watch_time),
       "payment_failures": int(failures),
       "customer_support_calls": int(calls)
   }
  
   try:
       # 1. Direct Prediction
       prediction = predict(payload_data)
       prob_value = float(prediction["probability"])
       
       # 2. Direct SHAP Explanation
       factors = []
       processed = preprocess_input(payload_data, encoders, feature_names)
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

       # 3. Format Verdict & Factors
       risk = prediction.get("risk_level", "N/A")
       risk_icon = "🔴" if risk == "HIGH" else "🟡" if risk == "MODERATE" else "🟢"
       summary = f"{risk_icon} Risk: {risk}\nProbability: {prob_value:.2%}"
       
       factors_text = "\n".join([f"{'🔺' if f['direction']=='increase' else '🔹'} {f['feature'].replace('_',' ').title()}: {f['impact_pct']}% impact" for f in factors])

       # 4. Direct Recommendations
       recom = generate_recommendations(prob_value, payload_data, top_factors=factors)
       actions = recom.get("recommended_actions", [])
       formatted_actions = "\n".join([f"✅ {a}" for a in actions]) if actions else "No specific actions."

       return summary, factors_text, formatted_actions
      
   except Exception as e:
       return f"❌ Logic Error: {str(e)}", "Check terminal logs", "Check terminal logs"

# ... keep the rest of your render_predict_page() ...

def render_predict_page():
   with gr.Column() as page:
       gr.Markdown("# 🔍 Customer Engagement & Churn Analysis")
      
       with gr.Row():
           with gr.Column(variant="panel"):
               gr.Markdown("### 👤 Profile")
               age = gr.Slider(18, 80, value=25, label="Age")
               gender = gr.Dropdown(["Male", "Female"], value="Male", label="Gender")
               sub_type = gr.Dropdown(["Basic", "Standard", "Premium"], value="Basic", label="Subscription")
               charges = gr.Number(value=20.0, label="Monthly Charges")
               tenure = gr.Slider(1, 72, value=12, label="Tenure (Months)")
              
               gr.Markdown("### 🖱️ Behavior")
               login_freq = gr.Slider(0, 30, value=10, label="Login Freq")
               last_login = gr.Slider(0, 60, value=5, label="Days Since Login")
               watch_time = gr.Number(value=10.0, label="Watch Time")
               failures = gr.Slider(0, 10, value=0, label="Payment Failures")
               calls = gr.Slider(0, 10, value=0, label="Support Calls")
              
               btn = gr.Button("Analyze Risk", variant="primary")


           with gr.Column():
               out_sum = gr.Textbox(label="Verdict", interactive=False)
               out_fac = gr.TextArea(label="Top Risk Drivers (SHAP)", interactive=False)
               out_rec = gr.TextArea(label="Recommended Retention Strategy", interactive=False, lines=10)


       btn.click(
           fn=get_prediction,
           inputs=[age, gender, sub_type, charges, tenure, login_freq, last_login, watch_time, failures, calls],
           outputs=[out_sum, out_fac, out_rec]
       )
   return page
