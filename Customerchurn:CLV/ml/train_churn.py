import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Import from your modular files
from customer_features import build_churn_features
from evaluate import evaluate_churn
from mock_data import get_mock_data

def train_and_save_churn_model():
    print("Loading mock data...")
    users_df, transactions_df = get_mock_data()

    X, y, features = build_churn_features(users_df, transactions_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Baseline: Logistic Regression
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    baseline.fit(X_train, y_train)
    base_recall, base_auc = evaluate_churn(y_test, baseline.predict(X_test), baseline.predict_proba(X_test)[:, 1])
    print(f"Baseline Logistic Regression -> Recall: {base_recall:.4f} | ROC-AUC: {base_auc:.4f}")

    # 2. Optimized Model: XGBClassifier
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
        eval_metric="logloss", random_state=42
    )
    model.fit(X_train, y_train)
    
    recall, roc_auc = evaluate_churn(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    print(f"Optimized XGBClassifier -> Recall: {recall:.4f} | ROC-AUC: {roc_auc:.4f}")

    # --- NEW FOLDER SAVING LOGIC ---
    # Get the directory of this script (ml folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and define the models folder path
    models_dir = os.path.join(script_dir, "..", "models")
    # Automatically create the models folder if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Define the exact file path
    save_path = os.path.join(models_dir, "churn_model.joblib")

    # Serialize Artifact Contract
    joblib.dump({
        "model": model,
        "features": features,
        "model_name": "XGBClassifier",
        "recall": recall,
        "roc_auc": roc_auc
    }, save_path)
    print(f"Saved model to {save_path} successfully.")

if __name__ == "__main__":
    train_and_save_churn_model()