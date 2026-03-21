import pandas as pd
import numpy as np

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/netflix_customer_churn.csv")  # <-- change filename if needed

# -----------------------------
# 2. Clean Column Names
# -----------------------------
df.columns = df.columns.str.strip().str.lower()

# -----------------------------
# 3. Rename Columns (Match Model Format)
# -----------------------------
df = df.rename(columns={
    "monthly_fee": "monthly_charges",
    "churned": "churn"
})

# -----------------------------
# 4. Create watch_time (IMPORTANT)
# -----------------------------
# Convert daily watch time → monthly
df["watch_time"] = df["avg_watch_time_per_day"] * 30

# -----------------------------
# 5. Feature Engineering
# -----------------------------

# Engagement-based feature
df["login_frequency"] = (df["watch_time"] * 0.5).astype(int)

# Simulated realistic features
df["payment_failures"] = np.random.randint(0, 3, size=len(df))
df["customer_support_calls"] = np.random.randint(0, 5, size=len(df))
df["tenure_in_months"] = np.random.randint(1, 72, size=len(df))

# -----------------------------
# 6. Debug Check (VERY IMPORTANT)
# -----------------------------
print("Columns after processing:\n", df.columns.tolist())

# -----------------------------
# 7. Select Final Columns
# -----------------------------
final_cols = [
    "age",
    "gender",
    "subscription_type",
    "monthly_charges",
    "tenure_in_months",
    "login_frequency",
    "last_login_days",
    "watch_time",
    "payment_failures",
    "customer_support_calls",
    "churn"
]

df = df[final_cols]

print("\nFinal columns:\n", df.columns.tolist())
# -----------------------------
# 8. Save Processed Data
# -----------------------------
df.to_csv("data/processed_customers.csv", index=False)

print("\n✅ Preprocessing completed successfully!")
print("Saved to: data/processed_customers.csv")