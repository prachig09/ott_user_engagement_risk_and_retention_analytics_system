"""
Enhanced Model Training Script
Trains and compares multiple ML models for customer churn prediction.
Uses the Kaggle Telco Customer Churn dataset.

Enhancements over baseline:
  - SMOTE oversampling to handle class imbalance (applied to training set only)
  - Class-weight balancing for models that support it
  - OneHotEncoding for multi-class categoricals (no false ordinality)
  - Feature engineering (num_services, avg_charge_per_month, tenure_group)
  - Hyperparameter tuning via RandomizedSearchCV
  - 5-fold stratified cross-validation
  - 5 algorithms: LR, RF, Gradient Boosting, Naive Bayes, XGBoost
  - Best model selected by ROC-AUC (not accuracy)
  - Wilcoxon Signed-Rank Test for statistical significance (10-fold paired CV)
  - ROI/Profit analysis: (TP * $450) - (FP * $50)
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
from scipy.stats import wilcoxon
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


def load_and_preprocess_data(data_path):
    """
    Load and preprocess the Telco Customer Churn dataset.

    Improvements:
      - Feature engineering (num_services, avg_charge_per_month)
      - OneHotEncoding for multi-class categoricals
      - LabelEncoder only for true binary columns

    Returns:
        X_scaled, y, encoders, feature_names
    """
    print("\n[1/6] Loading data...")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} records")

    # Drop customerID
    df = df.drop('customerID', axis=1)

    # Clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']
    df.loc[df['TotalCharges'].isna(), 'TotalCharges'] = 0.0
    print(f"   Cleaned TotalCharges ({mask.sum()} blank values imputed)")

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    churn_count = df['Churn'].sum()
    stay_count = len(df) - churn_count
    print(f"   Class distribution: Stay={stay_count} ({stay_count/len(df)*100:.1f}%), "
          f"Churn={churn_count} ({churn_count/len(df)*100:.1f}%)")

    # ── Feature Engineering (before encoding categoricals) ──────────
    print("\n[2/6] Feature engineering...")

    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['num_services'] = df[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)

    df['avg_charge_per_month'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

    # Tenure group: bucket tenure into bands (captures non-linear churn risk)
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                                labels=['0-12', '13-24', '25-48', '49-72'],
                                include_lowest=True)

    print(f"   Added 'num_services' (range: {df['num_services'].min()}-{df['num_services'].max()})")
    print(f"   Added 'avg_charge_per_month' (mean: ${df['avg_charge_per_month'].mean():.2f})")
    print(f"   Added 'tenure_group' (4 bands)")

    # ── Encoding ────────────────────────────────────────────────────
    print("\n[3/6] Encoding features...")
    encoders = {}

    # Binary columns → LabelEncoder (0/1 is appropriate)
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"   {col} (binary): {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Multi-class columns → OneHotEncoding (no false ordinality)
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod', 'tenure_group']

    print(f"\n   OneHot encoding {len(multi_cols)} multi-class features...")
    df = pd.get_dummies(df, columns=multi_cols, drop_first=False, dtype=int)

    # SeniorCitizen is already 0/1

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    feature_names = X.columns.tolist()
    print(f"   Total features after encoding: {len(feature_names)}")

    # ── Scaling ─────────────────────────────────────────────────────
    print("\n[4/6] Scaling features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    encoders['scaler'] = scaler

    # Store metadata for prediction & explanation
    encoders['_binary_cols'] = binary_cols
    encoders['_multi_cols'] = multi_cols
    encoders['_all_categorical'] = binary_cols          # only binary use LabelEncoder
    encoders['_feature_names'] = feature_names
    encoders['_engineered'] = ['num_services', 'avg_charge_per_month', 'tenure_group']

    # Reverse map: dummy column → parent column (for explain.py)
    dummy_to_parent = {}
    for col in multi_cols:
        for fname in feature_names:
            if fname.startswith(f"{col}_"):
                dummy_to_parent[fname] = col
    encoders['_dummy_to_parent'] = dummy_to_parent

    return X_scaled.values, y, encoders, feature_names


# ────────────────────────────────────────────────────────────────────
#  Training with tuning
# ────────────────────────────────────────────────────────────────────

def tune_and_train_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train 5 models with CORRECT SMOTE handling (no data leakage).

    CRITICAL: SMOTE is applied correctly:
      1. Train/Test split done BEFORE this function (in main).
      2. During CV tuning: imblearn Pipeline wraps SMOTE + estimator together,
         so SMOTE is applied ONLY to each fold's training portion.
         The CV validation fold NEVER sees synthetic samples.
      3. Final model: SMOTE applied to full training set, then model trained.
      4. Test set: NEVER touched by SMOTE — honest evaluation.

    This prevents the classic SMOTE data leakage trap where synthetic
    samples from one CV fold leak into the validation fold.
    """
    print("\n[5/6] Training models (SMOTE applied correctly inside CV)...")
    print(f"   Training set: Stay={int((y_train==0).sum())}, Churn={int((y_train==1).sum())}")
    print(f"   Test set:     Stay={int((y_test==0).sum())},  Churn={int((y_test==1).sum())}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Churn ratio for XGBoost scale_pos_weight
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_ratio = neg_count / pos_count
    print(f"   Class imbalance ratio: {scale_ratio:.2f}:1 (negative:positive)")

    # ── Define search spaces ────────────────────────────────────────
    # Each config uses an imblearn Pipeline: SMOTE → Estimator
    # RandomizedSearchCV sees this pipeline, so SMOTE runs INSIDE each fold.
    smote = SMOTE(random_state=42, k_neighbors=5)

    configs = {
        'Logistic Regression': {
            'pipeline': ImbPipeline([
                ('smote', smote),
                ('clf', LogisticRegression(class_weight='balanced',
                                          max_iter=2000, random_state=42))
            ]),
            'params': {
                'clf__C': [0.01, 0.1, 0.5, 1, 5, 10, 50],
                'clf__penalty': ['l1', 'l2'],
                'clf__solver': ['liblinear', 'saga']
            },
            'n_iter': 20
        },
        'Random Forest': {
            'pipeline': ImbPipeline([
                ('smote', smote),
                ('clf', RandomForestClassifier(class_weight='balanced',
                                              random_state=42, n_jobs=-1))
            ]),
            'params': {
                'clf__n_estimators': [100, 200, 300, 500],
                'clf__max_depth': [5, 10, 15, 20, None],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4]
            },
            'n_iter': 30
        },
        'Gradient Boosting': {
            'pipeline': ImbPipeline([
                ('smote', smote),
                ('clf', GradientBoostingClassifier(random_state=42))
            ]),
            'params': {
                'clf__n_estimators': [100, 200, 300, 500],
                'clf__max_depth': [3, 5, 7],
                'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'clf__subsample': [0.7, 0.8, 0.9, 1.0],
                'clf__min_samples_split': [2, 5, 10]
            },
            'n_iter': 40
        },
        'XGBoost': {
            'pipeline': ImbPipeline([
                ('smote', smote),
                ('clf', XGBClassifier(
                    scale_pos_weight=scale_ratio,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            'params': {
                'clf__n_estimators': [100, 200, 300, 500],
                'clf__max_depth': [3, 5, 7, 9],
                'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'clf__subsample': [0.7, 0.8, 0.9, 1.0],
                'clf__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'clf__min_child_weight': [1, 3, 5],
                'clf__gamma': [0, 0.1, 0.3, 0.5],
                'clf__reg_alpha': [0, 0.01, 0.1],
                'clf__reg_lambda': [1, 1.5, 2]
            },
            'n_iter': 50
        },
        'Naive Bayes': {
            'pipeline': ImbPipeline([
                ('smote', smote),
                ('clf', GaussianNB())
            ]),
            'params': {
                'clf__var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            },
            'n_iter': 7
        }
    }

    results = {}
    best_auc = 0
    best_model_name = None
    best_model = None

    for name, cfg in configs.items():
        print(f"\n   ┌─ Tuning {name}...")
        print(f"   │  (SMOTE applied INSIDE each CV fold — no data leakage)")

        search = RandomizedSearchCV(
            estimator=cfg['pipeline'],          # Pipeline = SMOTE + model
            param_distributions=cfg['params'],
            n_iter=cfg['n_iter'],
            cv=cv,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            error_score='raise'
        )
        # Fit on ORIGINAL training data — SMOTE happens inside each CV fold
        search.fit(X_train, y_train)

        best_params = {k.replace('clf__', ''): v for k, v in search.best_params_.items()}
        print(f"   │  Best params: {best_params}")
        print(f"   │  Best CV ROC-AUC: {search.best_score_:.4f}  (honest — no leakage)")

        # ── Final model: SMOTE on full training set, then train ─────
        # This is correct: test set is completely separate
        smote_final = SMOTE(random_state=42, k_neighbors=5)
        X_train_sm, y_train_sm = smote_final.fit_resample(X_train, y_train)

        # Extract the best classifier (not pipeline) with tuned params
        best_clf = search.best_estimator_.named_steps['clf']

        # Clone with best params and retrain on SMOTE'd full training set
        clf_class = type(best_clf)
        clf_params = best_clf.get_params()
        final_model = clf_class(**clf_params)
        final_model.fit(X_train_sm, y_train_sm)

        # ── Evaluate on ORIGINAL test set (never seen SMOTE) ────────
        y_pred = final_model.predict(X_test)
        y_prob = final_model.predict_proba(X_test)[:, 1]

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_prob)

        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': search.best_score_,        # honest CV from pipeline
            'cv_std': search.cv_results_['std_test_score'][search.best_index_],
            'best_params': best_params,
            'model': final_model,
            'confusion_matrix': cm
        }

        print(f"   │  Test Accuracy:  {accuracy:.4f}")
        print(f"   │  Test Precision: {precision:.4f}")
        print(f"   │  Test Recall:    {recall:.4f}")
        print(f"   │  Test F1-Score:  {f1:.4f}")
        print(f"   │  Test ROC-AUC:   {roc_auc:.4f}")
        print(f"   └─ CV ROC-AUC:     {search.best_score_:.4f} ± "
              f"{search.cv_results_['std_test_score'][search.best_index_]:.4f}")

        # Track best by ROC-AUC
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_name = name
            best_model = final_model

    print(f"\n   ★ BEST MODEL: {best_model_name} (ROC-AUC = {best_auc:.4f})")

    return best_model, best_model_name, results


# ────────────────────────────────────────────────────────────────────
#  Statistical Significance & ROI
# ────────────────────────────────────────────────────────────────────

def run_statistical_tests(X_train, y_train, results, best_model_name):
    """
    Run 10-fold paired CV and Wilcoxon Signed-Rank Tests.

    Uses a SEPARATE 10-fold CV (not the 5-fold used for tuning) to collect
    per-fold AUC scores for all models, then runs pairwise Wilcoxon tests
    comparing the best model against each baseline.

    Why 10-fold? With only 5 folds, the minimum achievable two-sided p-value
    is 1/2^5 = 0.03125 (one-sided) or 0.0625 (two-sided), which may exceed
    the 0.05 significance threshold. 10 folds provide more statistical power.
    """
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("(10-Fold Paired CV + Wilcoxon Signed-Rank Test)")
    print("=" * 60)

    # Shared 10-fold CV — same folds for ALL models (required for paired test)
    cv_stat = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42, k_neighbors=5)

    fold_scores = {}

    for name, res in results.items():
        # Build pipeline with tuned model for honest 10-fold evaluation
        pipeline = ImbPipeline([
            ('smote', smote),
            ('clf', res['model'])  # Already has best hyperparameters
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv_stat, scoring='roc_auc', n_jobs=1
        )
        fold_scores[name] = scores
        print(f"\n   {name}:")
        print(f"     10-Fold AUC scores: {[f'{s:.4f}' for s in scores]}")
        print(f"     Mean: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Pairwise Wilcoxon tests: best model vs each other
    best_scores = fold_scores[best_model_name]

    print(f"\n{'─' * 60}")
    print(f"Wilcoxon Signed-Rank Test: {best_model_name} vs Others")
    print(f"H1: {best_model_name} has higher ROC-AUC (one-sided, a=0.05)")
    print(f"{'─' * 60}")

    header = "{:<25} {:>10} {:>10} {:>12}".format(
        "Comparison", "D Mean", "p-value", "Significant?"
    )
    print(header)
    print("-" * 60)

    wilcoxon_results = {}

    for name, scores in fold_scores.items():
        if name == best_model_name:
            wilcoxon_results[name] = {'p_value': None, 'significant': True, 'delta': 0.0}
            continue

        delta = best_scores.mean() - scores.mean()

        try:
            stat, p_value = wilcoxon(best_scores, scores, alternative='greater')
            significant = p_value < 0.05
            sig_marker = "YES" if significant else "NO"
            print("{:<25} {:>+10.4f} {:>10.4f} {:>12}".format(
                f"vs {name}", delta, p_value, sig_marker
            ))
        except ValueError:
            # All differences are zero (identical scores)
            p_value = 1.0
            significant = False
            print("{:<25} {:>+10.4f} {:>10} {:>12}".format(
                f"vs {name}", delta, "N/A", "NO"
            ))

        wilcoxon_results[name] = {
            'p_value': p_value,
            'significant': significant,
            'delta': delta
        }

    # Summary
    sig_count = sum(1 for v in wilcoxon_results.values()
                    if v['significant'] and v['p_value'] is not None)
    total = len(wilcoxon_results) - 1  # exclude self

    print(f"\n   RESULT: {best_model_name} is statistically significantly better than "
          f"{sig_count}/{total} other models (a=0.05)")

    return wilcoxon_results, fold_scores


def calculate_roi(results):
    """
    Calculate ROI (profit) for each model.

    Formula: ROI = (TP x $450) - (FP x $50)

    - TP x $450: Revenue saved by correctly identifying and retaining a churner
    - FP x $50: Cost of offering unnecessary retention incentives to non-churners

    This translates model performance into business value.
    """
    print("\n" + "=" * 60)
    print("ROI / PROFIT ANALYSIS")
    print("Formula: ROI = (TP x $450) - (FP x $50)")
    print("=" * 60)

    header = "{:<25} {:>6} {:>6} {:>6} {:>6} {:>12}".format(
        "Model", "TP", "FP", "TN", "FN", "ROI ($)"
    )
    print(header)
    print("-" * 67)

    roi_results = {}
    best_roi = float('-inf')
    best_roi_model = None

    for name, res in results.items():
        cm = res.get('confusion_matrix')
        if cm is None:
            continue

        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        revenue_saved = tp * 450      # Correctly caught churners
        campaign_cost = fp * 50       # False alarms — wasted retention cost
        roi = revenue_saved - campaign_cost

        roi_results[name] = {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'revenue_saved': revenue_saved,
            'campaign_cost': campaign_cost,
            'roi': roi
        }

        if roi > best_roi:
            best_roi = roi
            best_roi_model = name

        print("{:<25} {:>6d} {:>6d} {:>6d} {:>6d} {:>12}".format(
            name, tp, fp, tn, fn, f"${roi:,}"
        ))

    # Summary
    print(f"\n   BEST ROI: {best_roi_model} — ${best_roi:,}")
    print(f"     Revenue saved (TP x $450):  ${roi_results[best_roi_model]['revenue_saved']:,}")
    print(f"     Campaign cost (FP x $50):   ${roi_results[best_roi_model]['campaign_cost']:,}")
    print(f"     Net profit:                 ${best_roi:,}")

    return roi_results


# ────────────────────────────────────────────────────────────────────
#  Save & Report
# ────────────────────────────────────────────────────────────────────

def save_model_and_encoder(model, encoders, feature_names, model_dir):
    """Save the trained model, encoders, and feature names."""
    print("\n[6/6] Saving model and encoders...")
    os.makedirs(model_dir, exist_ok=True)

    model_path   = os.path.join(model_dir, 'churn_model.pkl')
    encoder_path = os.path.join(model_dir, 'encoder.pkl')
    feature_path = os.path.join(model_dir, 'feature_names.pkl')

    joblib.dump(model, model_path)
    joblib.dump(encoders, encoder_path)
    joblib.dump(feature_names, feature_path)

    print(f"   Model saved to: {model_path}")
    print(f"   Encoders saved to: {encoder_path}")
    print(f"   Feature names saved to: {feature_path}")


def print_detailed_report(model, X_test, y_test, model_name, results):
    """Print detailed classification report, confusion matrix, and cross-val."""
    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print(f"DETAILED REPORT — {model_name}")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Stay  Churn")
    print(f"   Actual Stay    {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"   Actual Churn   {cm[1][0]:4d}  {cm[1][1]:4d}")

    r = results[model_name]
    print(f"\n5-Fold Cross-Validation ROC-AUC: {r['cv_mean']:.4f} ± {r['cv_std']:.4f}")
    print(f"Best hyperparameters: {r['best_params']}")


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ENHANCED CUSTOMER CHURN MODEL TRAINING")
    print("Telco Customer Churn Dataset (Kaggle)")
    print("SMOTE + XGBoost Enhanced Pipeline")
    print("=" * 60)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Locate dataset
    data_path = os.path.join(project_dir, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(project_dir, 'data', 'customers.csv')
    model_dir = os.path.join(project_dir, 'model')

    if not os.path.exists(data_path):
        print(f"\n[ERROR] Dataset not found at: {data_path}")
        return

    # Load & preprocess
    X, y, encoders, feature_names = load_and_preprocess_data(data_path)

    # Split (stratified)
    print("\n   Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")

    # Train with tuning
    best_model, best_model_name, results = tune_and_train_models(
        X_train, X_test, y_train, y_test, feature_names
    )

    # Detailed report
    print_detailed_report(best_model, X_test, y_test, best_model_name, results)

    # Save
    save_model_and_encoder(best_model, encoders, feature_names, model_dir)

    # ── Summary table ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY  (sorted by ROC-AUC)")
    print("=" * 60)
    header = "{:<25} {:>8} {:>8} {:>8} {:>8} {:>8} {:>12}".format(
        "Model", "Acc", "Prec", "Recall", "F1", "AUC", "CV AUC"
    )
    print(header)
    print("-" * 87)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    for name, m in sorted_results:
        marker = " ★" if name == best_model_name else ""
        cv_str = f"{m['cv_mean']:.4f}±{m['cv_std']:.3f}"
        print("{:<25} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>12s}{}".format(
            name, m['accuracy'], m['precision'], m['recall'],
            m['f1_score'], m['roc_auc'], cv_str, marker
        ))

    # ── ROI / Profit Analysis ──────────────────────────────────────
    roi_results = calculate_roi(results)

    # ── Statistical Significance Tests ─────────────────────────────
    wilcoxon_results, fold_scores = run_statistical_tests(
        X_train, y_train, results, best_model_name
    )

    # ── Save extended results ──────────────────────────────────────
    extended_results = {
        'wilcoxon': wilcoxon_results,
        'roi': roi_results,
        'fold_scores': {name: scores.tolist() for name, scores in fold_scores.items()},
        'best_model_name': best_model_name,
    }
    extended_path = os.path.join(model_dir, 'extended_results.pkl')
    joblib.dump(extended_results, extended_path)
    print(f"\n   Extended results saved to: {extended_path}")

    print("\n" + "=" * 60)
    print("[OK] Enhanced training complete! Model saved successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()