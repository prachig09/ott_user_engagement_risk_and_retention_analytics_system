import pandas as pd
import numpy as np
import os
import joblib
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

mlflow.set_experiment("OTT Churn Prediction")

warnings.filterwarnings("ignore")


# =========================
# LOAD + PREPROCESS DATA
# =========================
def load_and_preprocess_data(data_path):

    print("\n[1/4] Loading data...")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} records")

    # Drop ID if exists
    if "customer_id" in df.columns:
        df = df.drop("customer_id", axis=1)

    # Target
    y = df["churn"]
    X = df.drop("churn", axis=1)

    print(f"   Class distribution → Stay: {(y==0).sum()}, Churn: {(y==1).sum()}")

    # Encode categorical
    print("\n[2/4] Encoding features...")
    categorical_cols = ["gender", "subscription_type"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    feature_names = X.columns.tolist()
    print(f"   Total features after encoding: {len(feature_names)}")

    # Scale
    print("\n[3/4] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoders = {
        "scaler": scaler,
        "feature_names": feature_names
    }

    return X_scaled, y, encoders, feature_names


# =========================
# TRAIN MODELS
# =========================
def train_models(X_train, X_test, y_train, y_test):

    print("\n[4/4] Training models with SMOTE...")

    smote = SMOTE(random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
        "Naive Bayes": GaussianNB()
    }

    results = {}
    best_model = None
    best_name = None
    best_auc = 0

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            print(f"\n🔹 Training {name}...")

            pipeline = ImbPipeline([
                ("smote", smote),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            # 🔥 LOG METRICS
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)

            # 🔥 LOG MODEL
            mlflow.sklearn.log_model(pipeline, name)

            results[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc,
                "model": pipeline,
                "cm": confusion_matrix(y_test, y_pred)
            }

            print(f"   AUC: {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_model = pipeline
                best_name = name

    print(f"\n✅ Best Model: {best_name} (AUC: {best_auc:.4f})")

    return best_model, best_name, results


# =========================
# ROI CALCULATION
# =========================
def calculate_roi(results):

    print("\n💰 ROI ANALYSIS")

    for name, res in results.items():
        tn, fp, fn, tp = res["cm"].ravel()

        roi = (tp * 450) - (fp * 50)

        print(f"{name}: ROI = ${roi}")


# =========================
# SAVE MODEL
# =========================
def save_model(model, encoders, feature_names):

    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/churn_model.pkl")
    joblib.dump(encoders, "model/encoder.pkl")
    joblib.dump(feature_names, "model/feature_names.pkl")

    print("\n💾 Model saved successfully!")


# =========================
# MAIN
# =========================
def main():

    print("="*50)
    print("OTT CHURN MODEL TRAINING")
    print("="*50)

    data_path = "data/processed_customers.csv"

    X, y, encoders, feature_names = load_and_preprocess_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    best_model, best_name, results = train_models(X_train, X_test, y_train, y_test)

    calculate_roi(results)

    save_model(best_model, encoders, feature_names)

    print("\n🎯 Training Completed Successfully!")


if __name__ == "__main__":
    main()