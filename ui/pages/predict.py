import gradio as gr
import requests


API_URL = "http://0.0.0.0:8000/full"


def get_prediction(age, gender, sub_type, charges, tenure, login_freq, last_login, watch_time, failures, calls):
   payload = {
       "data": {
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
   }
  
   try:
       response = requests.post(API_URL, json=payload, timeout=25)
       res_data = response.json()
      
       # --- DEBUG: CHECK YOUR TERMINAL ---
       print("\n--- API DEBUG OUTPUT ---")
       print(res_data)
      
       # 1. Parse Prediction
       pred = res_data.get("prediction", {})
       risk = pred.get("risk_level", "N/A")
       prob = pred.get("probability", 0)
       risk_icon = "🔴" if risk == "HIGH" else "🟡" if risk == "MODERATE" else "🟢"
      
       summary = f"{risk_icon} Risk: {risk}\nProbability: {prob:.2%}"
      
       # Inside ui/pages/predict.py -> get_prediction function


       # 2. Parse SHAP Factors with Percentage
       factors_list = res_data.get("explanation", [])
       formatted_factors = []
      
       for f in factors_list:
           feat = f['feature'].replace("_", " ").title() # Clean name: "watch_time" -> "Watch Time"
           direction = "🔺" if f['direction'] == "increase" else "🔹"
           pct = f['impact_pct']
          
           # Example output: "🔺 Watch Time: +15.4% impact"
           formatted_factors.append(f"{direction} {feat}: {pct}% contribution")
      
       factors_text = "\n".join(formatted_factors) if formatted_factors else "No significant factors found."
      
       # 3. Parse Recommendations (Crucial fix here!)
       recom = res_data.get("recommendation", {})
       actions = recom.get("recommended_actions", [])
      
       if actions:
           formatted_actions = "\n".join([f"✅ {a}" for a in actions])
       else:
           # If empty, let's see why
           formatted_actions = f"No actions returned. API Keys: {list(recom.keys())}"


       return summary, factors_text, formatted_actions
      
   except Exception as e:
       print(f"UI ERROR: {e}")
       return f"❌ Connection Error: {str(e)}", "Check API terminal", "Check API terminal"


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
