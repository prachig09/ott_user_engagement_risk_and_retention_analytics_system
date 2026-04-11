"""
OTT Recommendation Engine
Generates retention strategies based on churn prediction.
"""




def get_risk_level(prob):
   if prob >= 0.7:
       return "HIGH"
   elif prob >= 0.4:
       return "MODERATE"
   return "LOW"




def generate_recommendations(churn_probability, customer_data, top_factors=None):
   """
   Generate actionable business recommendations.
   """


   risk = get_risk_level(churn_probability)


   recommendations = []


   # 🎯 BASE ACTIONS
   if risk == "HIGH":
       recommendations.append("Offer 30% discount on next subscription")
       recommendations.append("Send personalized content recommendations immediately")
       recommendations.append("Push notification: 'We miss you! New shows added'")
   elif risk == "MODERATE":
       recommendations.append("Send weekly content recommendations email")
       recommendations.append("Offer limited-time discount (10-15%)")
   else:
       recommendations.append("Maintain regular engagement")


   # 🎯 FEATURE-BASED ACTIONS


   if customer_data.get("watch_time", 0) < 5:
       recommendations.append("Recommend trending shows to increase watch time")


   if customer_data.get("last_login_days", 0) > 7:
       recommendations.append("Send re-engagement notification/email")


   if customer_data.get("payment_failures", 0) > 0:
       recommendations.append("Prompt user to update payment details")


   if customer_data.get("login_frequency", 0) < 2:
       recommendations.append("Send app reminders to increase usage")


   if customer_data.get("subscription_type") == "Basic":
       recommendations.append("Offer upgrade to Premium at discounted rate")


   if customer_data.get("customer_support_calls", 0) > 2:
       recommendations.append("Assign customer success agent for follow-up")


   # 🔥 PRIORITIZE TOP FACTORS (if SHAP available)
   if top_factors:
       important = [f['feature'] for f in top_factors]


       if "watch_time" in important:
           recommendations.insert(0, "CRITICAL: Increase engagement via personalized recommendations")


       if "last_login_days" in important:
           recommendations.insert(0, "CRITICAL: User inactive — immediate re-engagement required")


   # Remove duplicates
   recommendations = list(dict.fromkeys(recommendations))


   return {
    "risk_level": risk,
    "churn_probability": churn_probability,
    "recommended_actions": recommendations[:5]  # <-- The UI is looking for this exact key
}




# ✅ TEST
def main():
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


   rec = generate_recommendations(0.68, sample_customer, [
       {"feature": "watch_time"},
       {"feature": "last_login_days"}
   ])


   print("\n📢 RECOMMENDATIONS")
   print("=" * 40)
   print("Risk:", rec["risk_level"])
   print("Probability:", rec["churn_probability"])


   for i, r in enumerate(rec["recommended_actions"], 1):
       print(f"{i}. {r}")




if __name__ == "__main__":
   main()
