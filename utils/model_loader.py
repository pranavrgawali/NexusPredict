"""
NexusPredict — ML Model Loader for Customer Churn & CLV
Loads serialized .joblib artifacts and runs inference.
Falls back gracefully when models are not found.
"""

import os
import numpy as np
import pandas as pd

# ── Try loading joblib ──
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# ── Model paths (relative to project root) ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHURN_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "churn_model.joblib")
CLV_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "clv_model.joblib")

# ── Cached model references ──
_churn_cache = None
_clv_cache = None


# ═══════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════

def load_churn_model():
    """Load the churn classification model artifact.
    Returns dict with keys: model, features, model_name, recall, roc_auc
    Returns None if model file is missing or joblib not installed.
    """
    global _churn_cache
    if _churn_cache is not None:
        return _churn_cache
    if not JOBLIB_AVAILABLE or not os.path.exists(CHURN_MODEL_PATH):
        return None
    try:
        _churn_cache = joblib.load(CHURN_MODEL_PATH)
        return _churn_cache
    except Exception:
        return None


def load_clv_model():
    """Load the CLV regression model artifact.
    Returns dict with keys: model, features, model_name, mae, rmse
    Returns None if model file is missing or joblib not installed.
    """
    global _clv_cache
    if _clv_cache is not None:
        return _clv_cache
    if not JOBLIB_AVAILABLE or not os.path.exists(CLV_MODEL_PATH):
        return None
    try:
        _clv_cache = joblib.load(CLV_MODEL_PATH)
        return _clv_cache
    except Exception:
        return None


def is_churn_model_available():
    """Check if the churn model can be loaded."""
    return load_churn_model() is not None


def is_clv_model_available():
    """Check if the CLV model can be loaded."""
    return load_clv_model() is not None


# ═══════════════════════════════════════════════════════════
# Metrics Extractors
# ═══════════════════════════════════════════════════════════

def get_churn_model_metrics():
    """Extract stored evaluation metrics from the churn model artifact."""
    artifact = load_churn_model()
    if artifact is None:
        return None
    return {
        "model_name": artifact.get("model_name", "Unknown"),
        "recall": artifact.get("recall", 0.0),
        "roc_auc": artifact.get("roc_auc", 0.0),
    }


def get_clv_model_metrics():
    """Extract stored evaluation metrics from the CLV model artifact."""
    artifact = load_clv_model()
    if artifact is None:
        return None
    return {
        "model_name": artifact.get("model_name", "Unknown"),
        "mae": artifact.get("mae", 0.0),
        "rmse": artifact.get("rmse", 0.0),
    }


# ═══════════════════════════════════════════════════════════
# Inference — Churn
# ═══════════════════════════════════════════════════════════

def predict_churn(customer_df):
    """Run churn prediction on a DataFrame of customer features.

    Args:
        customer_df: DataFrame with columns matching the model's expected features
                     (age, login_frequency_per_month, support_tickets_raised,
                      is_active_subscriber, transaction_count, total_spend,
                      average_order_value, recency_days)

    Returns:
        DataFrame with added columns: Risk Score (%), Risk Label
        or None if model is unavailable.
    """
    artifact = load_churn_model()
    if artifact is None:
        return None

    model = artifact["model"]
    features = artifact["features"]

    # Ensure all required feature columns exist
    missing = [f for f in features if f not in customer_df.columns]
    if missing:
        return None

    X = customer_df[features].copy()

    # Predict probabilities (churn probability = class 1)
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        return None

    risk_scores = np.clip(proba * 100, 1, 99)

    result = customer_df.copy()
    result["Risk Score (%)"] = np.round(risk_scores, 1)
    result["Risk Label"] = pd.cut(
        risk_scores, bins=[0, 30, 60, 100], labels=["Low", "Medium", "High"]
    )

    return result.sort_values("Risk Score (%)", ascending=False).reset_index(drop=True)


def get_churn_feature_importance_from_model():
    """Extract feature importances from the trained XGBoost model.
    Returns a DataFrame with Feature and Importance columns, or None.
    """
    artifact = load_churn_model()
    if artifact is None:
        return None

    model = artifact["model"]
    features = artifact["features"]

    try:
        importances = model.feature_importances_
    except AttributeError:
        return None

    # Prettify feature names for display
    name_map = {
        "age": "Customer Age",
        "login_frequency_per_month": "Login Frequency",
        "support_tickets_raised": "Support Tickets Raised",
        "is_active_subscriber": "Subscription Status",
        "transaction_count": "Transaction Count",
        "total_spend": "Total Spend (Lifetime)",
        "average_order_value": "Avg Order Value",
        "recency_days": "Days Since Last Purchase",
    }

    df = pd.DataFrame({
        "Feature": [name_map.get(f, f) for f in features],
        "Importance": np.round(importances, 4),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════
# Inference — CLV
# ═══════════════════════════════════════════════════════════

def predict_clv(customer_df):
    """Run CLV prediction on a DataFrame of customer features.

    Args:
        customer_df: DataFrame with columns matching the CLV model's features
                     (recency_days, transaction_count, total_spend,
                      average_order_value, login_frequency_per_month,
                      support_tickets_raised, is_active_subscriber)

    Returns:
        DataFrame with added columns: Predicted CLV (₹), Tier
        or None if model is unavailable.
    """
    artifact = load_clv_model()
    if artifact is None:
        return None

    model = artifact["model"]
    features = artifact["features"]

    missing = [f for f in features if f not in customer_df.columns]
    if missing:
        return None

    X = customer_df[features].copy()

    try:
        predicted = model.predict(X)
    except Exception:
        return None

    predicted = np.clip(predicted, 0, None)  # CLV can't be negative

    result = customer_df.copy()
    result["Predicted CLV (₹)"] = np.round(predicted, 2)

    # Assign tiers
    result["Tier"] = pd.cut(
        predicted,
        bins=[-np.inf, 500, 2000, 8000, np.inf],
        labels=["Bronze", "Silver", "Gold", "Platinum"],
    )

    return result.sort_values("Predicted CLV (₹)", ascending=False).reset_index(drop=True)


def get_clv_feature_importance_from_model():
    """Extract feature importances from the CLV regression model.
    Returns a DataFrame with Feature and Importance columns, or None.
    """
    artifact = load_clv_model()
    if artifact is None:
        return None

    model = artifact["model"]
    features = artifact["features"]

    try:
        importances = model.feature_importances_
    except AttributeError:
        return None

    name_map = {
        "recency_days": "Recency (Days)",
        "transaction_count": "Transaction Count",
        "total_spend": "Total Spend (Lifetime)",
        "average_order_value": "Avg Order Value",
        "login_frequency_per_month": "Login Frequency",
        "support_tickets_raised": "Support Tickets",
        "is_active_subscriber": "Subscription Status",
    }

    df = pd.DataFrame({
        "Feature": [name_map.get(f, f) for f in features],
        "Importance": np.round(importances, 4),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df
