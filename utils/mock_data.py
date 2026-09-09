"""
NexusPredict — Mock Data Generators
Realistic fake data for the frontend UI. Uses fixed seeds for reproducibility.
These functions will be replaced with actual MySQL/ML pipeline calls later.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ── Helper ──
COUNTRIES = ["United States", "United Kingdom", "Germany", "India", "Canada",
             "Australia", "France", "Brazil", "Japan", "Singapore"]

CATEGORIES = ["Electronics", "Clothing", "Home & Kitchen", "Books",
              "Sports & Outdoors", "Beauty", "Toys & Games", "Grocery"]

CHANNELS = ["Organic Search", "Paid Ads", "Social Media", "Email Marketing",
            "Direct", "Referral", "Affiliate"]

FIRST_NAMES = ["Aarav", "Sophia", "Liam", "Emma", "Noah", "Olivia", "James",
               "Mia", "Lucas", "Aria", "Ethan", "Chloe", "Rohan", "Zara",
               "Mason", "Lily", "Aiden", "Priya", "Elijah", "Hannah",
               "Benjamin", "Amara", "Leo", "Isla", "Jack", "Nora", "Ryan",
               "Freya", "Daniel", "Ananya", "Samuel", "Ava", "Henry", "Ruby",
               "Owen", "Grace", "Kai", "Sana", "Dylan", "Elena"]

LAST_NAMES = ["Sharma", "Smith", "Müller", "Patel", "Johnson", "Williams",
              "Brown", "Wilson", "Lee", "Anderson", "Garcia", "Chen",
              "Nakamura", "Singh", "Taylor", "Thomas", "Martinez", "Kumar",
              "Davis", "Kim"]


def _random_names(n):
    return [f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}" for _ in range(n)]


# ═══════════════════════════════════════════════════════════
# 1. EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════

def get_dashboard_kpis():
    """Key performance indicators for the executive overview."""
    return {
        "total_revenue": 2_847_392.50,
        "revenue_delta": 12.4,
        "active_users": 18_432,
        "users_delta": 8.7,
        "avg_order_value": 154.62,
        "aov_delta": 3.2,
        "monthly_orders": 6_721,
        "orders_delta": 15.1,
        "churn_rate": 14.3,
        "churn_delta": -2.1,
        "conversion_rate": 3.8,
        "conversion_delta": 0.6,
    }


def get_revenue_trend():
    """Monthly revenue trend for the past 12 months."""
    months = pd.date_range(end=datetime.now(), periods=12, freq='ME')
    base = 180_000
    trend = np.linspace(0, 60_000, 12)
    noise = np.random.normal(0, 12_000, 12)
    revenue = base + trend + noise
    revenue = np.maximum(revenue, 100_000)

    return pd.DataFrame({
        "Month": months,
        "Revenue": np.round(revenue, 2),
        "Target": np.round(base + trend + 10_000, 2),
    })


def get_user_acquisition():
    """User acquisition breakdown by channel."""
    users = [4200, 3800, 3100, 2600, 2200, 1500, 1032]
    return pd.DataFrame({
        "Channel": CHANNELS,
        "Users": users,
        "Percentage": np.round(np.array(users) / sum(users) * 100, 1)
    })


def get_category_sales():
    """Revenue breakdown by product category."""
    sales = [520000, 410000, 380000, 290000, 350000, 310000, 270000, 317392]
    return pd.DataFrame({
        "Category": CATEGORIES,
        "Sales": sales,
        "Percentage": np.round(np.array(sales) / sum(sales) * 100, 1)
    })


def get_recent_transactions(n=15):
    """Recent transaction records."""
    np.random.seed(99)
    now = datetime.now()
    return pd.DataFrame({
        "Transaction ID": [f"TXN-{np.random.randint(100000, 999999)}" for _ in range(n)],
        "Customer": _random_names(n),
        "Date": [(now - timedelta(hours=np.random.randint(1, 168))).strftime("%Y-%m-%d %H:%M") for _ in range(n)],
        "Amount (₹)": np.round(np.random.lognormal(4.2, 0.8, n), 2),
        "Items": np.random.randint(1, 8, n),
        "Country": np.random.choice(COUNTRIES, n),
    })


def get_monthly_stats():
    """Monthly aggregated stats for sparkline-type data."""
    months = pd.date_range(end=datetime.now(), periods=6, freq='ME')
    return pd.DataFrame({
        "Month": months,
        "Revenue": np.round(np.random.uniform(200000, 300000, 6), 2),
        "Users": np.random.randint(2500, 4000, 6),
        "Orders": np.random.randint(800, 1500, 6),
    })


# ═══════════════════════════════════════════════════════════
# RAW CUSTOMER DATA (for ML model inference)
# ═══════════════════════════════════════════════════════════

def get_raw_customer_data(n=200):
    """Generate raw customer records with the exact feature columns
    expected by the trained churn and CLV models.

    Returns a DataFrame ready for model_loader.predict_churn() and
    model_loader.predict_clv() inference.
    """
    np.random.seed(42)

    # Features matching the trained model contract
    ages = np.random.randint(18, 65, n)
    login_freq = np.random.randint(0, 20, n)
    support_tickets = np.random.randint(0, 6, n)
    is_active = np.random.choice([0, 1], n)
    tx_count = np.random.randint(0, 50, n)
    total_spend = np.round(np.random.lognormal(6, 1.2, n), 2)
    avg_order = np.where(tx_count > 0,
                         np.round(total_spend / np.maximum(tx_count, 1), 2),
                         0.0)
    recency = np.random.randint(1, 180, n)

    return pd.DataFrame({
        "User ID": [f"USR-{i+1001}" for i in range(n)],
        "Customer Name": _random_names(n),
        "Country": np.random.choice(COUNTRIES, n),
        # Model features (exact column names from training)
        "age": ages,
        "login_frequency_per_month": login_freq,
        "support_tickets_raised": support_tickets,
        "is_active_subscriber": is_active,
        "transaction_count": tx_count,
        "total_spend": total_spend,
        "average_order_value": avg_order,
        "recency_days": recency,
    })


# ═══════════════════════════════════════════════════════════
# 2. CHURN PREDICTIONS
# ═══════════════════════════════════════════════════════════

def get_churn_predictions(n=100):
    """Customer churn risk predictions with features."""
    np.random.seed(42)
    risk_scores = np.clip(np.random.beta(2, 5, n) * 100, 1, 99)
    # Skew some to high risk
    high_risk_idx = np.random.choice(n, size=25, replace=False)
    risk_scores[high_risk_idx] = np.random.uniform(70, 98, 25)

    risk_labels = pd.cut(risk_scores, bins=[0, 30, 60, 100],
                         labels=["Low", "Medium", "High"])

    days_since_purchase = np.where(risk_scores > 60,
                                   np.random.randint(30, 120, n),
                                   np.random.randint(1, 45, n))

    return pd.DataFrame({
        "User ID": [f"USR-{i+1001}" for i in range(n)],
        "Customer Name": _random_names(n),
        "Country": np.random.choice(COUNTRIES, n),
        "Age": np.random.randint(18, 65, n),
        "Days Since Last Purchase": days_since_purchase,
        "Login Frequency (monthly)": np.where(risk_scores > 60,
                                               np.random.randint(0, 4, n),
                                               np.random.randint(3, 20, n)),
        "Support Tickets": np.random.poisson(2, n),
        "Total Spend (₹)": np.round(np.random.lognormal(6, 1.2, n), 2),
        "Risk Score (%)": np.round(risk_scores, 1),
        "Risk Label": risk_labels,
    }).sort_values("Risk Score (%)", ascending=False).reset_index(drop=True)


def get_churn_feature_importance():
    """Feature importance from the churn model."""
    features = [
        "Days Since Last Purchase",
        "Login Frequency",
        "Support Tickets Raised",
        "Total Spend (Lifetime)",
        "Avg Order Value",
        "Account Age (Days)",
        "Items Purchased",
        "Subscription Status",
    ]
    importance = [0.28, 0.22, 0.15, 0.12, 0.09, 0.06, 0.05, 0.03]
    return pd.DataFrame({
        "Feature": features,
        "Importance": importance,
    })


def get_churn_summary():
    """Summary statistics for churn overview."""
    return {
        "total_customers": 18432,
        "high_risk": 2765,
        "medium_risk": 4608,
        "low_risk": 11059,
        "avg_risk_score": 34.2,
        "model_accuracy": 91.3,
        "roc_auc": 0.943,
        "recall": 0.887,
    }


# ═══════════════════════════════════════════════════════════
# 3. CLV (CUSTOMER LIFETIME VALUE)
# ═══════════════════════════════════════════════════════════

def get_clv_predictions(n=100):
    """Customer Lifetime Value predictions with RFM features."""
    np.random.seed(77)
    recency = np.random.randint(1, 180, n)
    frequency = np.random.randint(1, 50, n)
    monetary = np.round(np.random.lognormal(4.5, 1.0, n), 2)

    # CLV correlates with frequency and monetary
    base_clv = (frequency * monetary * 0.3) + np.random.normal(500, 200, n)
    predicted_clv = np.clip(np.round(base_clv, 2), 50, 50000)

    tiers = pd.cut(predicted_clv,
                    bins=[0, 500, 2000, 8000, 50001],
                    labels=["Bronze", "Silver", "Gold", "Platinum"])

    return pd.DataFrame({
        "User ID": [f"USR-{i+1001}" for i in range(n)],
        "Customer Name": _random_names(n),
        "Country": np.random.choice(COUNTRIES, n),
        "Recency (Days)": recency,
        "Frequency": frequency,
        "Avg Monetary (₹)": monetary,
        "Predicted CLV (₹)": predicted_clv,
        "Tier": tiers,
    }).sort_values("Predicted CLV (₹)", ascending=False).reset_index(drop=True)


def get_clv_metrics():
    """Model evaluation metrics for the CLV regressor."""
    return {
        "mae": 342.17,
        "rmse": 487.93,
        "r2_score": 0.847,
        "mape": 15.3,
    }


def get_clv_tier_summary():
    """Summary stats per CLV tier."""
    return pd.DataFrame({
        "Tier": ["Platinum", "Gold", "Silver", "Bronze"],
        "Count": [423, 2841, 8210, 6958],
        "Avg CLV (₹)": [12847.30, 4523.10, 1245.80, 287.40],
        "Total Revenue (₹)": [5434147.90, 12849146.10, 10228218.00, 1999369.20],
        "Color": ["#8b5cf6", "#f59e0b", "#94a3b8", "#d97706"],
    })


# ═══════════════════════════════════════════════════════════
# 4. DEMAND FORECASTING
# ═══════════════════════════════════════════════════════════

def get_forecast_data(category="Electronics"):
    """30-day demand forecast with historical data and confidence intervals."""
    np.random.seed(hash(category) % 2**31)

    # Historical (past 60 days)
    base_demand = {"Electronics": 150, "Clothing": 200, "Home & Kitchen": 120,
                   "Books": 80, "Sports & Outdoors": 90, "Beauty": 130,
                   "Toys & Games": 70, "Grocery": 250}

    base = base_demand.get(category, 100)
    today = datetime.now().date()
    hist_dates = [today - timedelta(days=i) for i in range(60, 0, -1)]

    # Weekly seasonality
    weekly = [1.0, 0.9, 0.85, 0.88, 0.95, 1.15, 1.2]
    hist_values = []
    for d in hist_dates:
        seasonal = weekly[d.weekday()]
        val = base * seasonal + np.random.normal(0, base * 0.15)
        hist_values.append(max(int(val), 5))

    # Forecast (next 30 days)
    forecast_dates = [today + timedelta(days=i) for i in range(1, 31)]
    forecast_values = []
    lower_bounds = []
    upper_bounds = []

    for i, d in enumerate(forecast_dates):
        seasonal = weekly[d.weekday()]
        growth = 1 + (i * 0.003)  # slight upward trend
        val = base * seasonal * growth + np.random.normal(0, base * 0.08)
        val = max(int(val), 5)
        margin = int(base * 0.2 * (1 + i * 0.02))  # widening confidence interval
        forecast_values.append(val)
        lower_bounds.append(max(val - margin, 0))
        upper_bounds.append(val + margin)

    hist_df = pd.DataFrame({
        "Date": hist_dates,
        "Units Sold": hist_values,
        "Type": "Historical",
    })

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Units Sold": forecast_values,
        "Lower Bound": lower_bounds,
        "Upper Bound": upper_bounds,
        "Type": "Forecast",
    })

    return hist_df, forecast_df


def get_inventory_status():
    """Current inventory status with restock alerts."""
    np.random.seed(55)
    stock = np.random.randint(50, 500, len(CATEGORIES))
    demand_30d = np.random.randint(200, 800, len(CATEGORIES))
    safety_stock = (demand_30d * 0.3).astype(int)
    restock_needed = stock < safety_stock

    return pd.DataFrame({
        "Category": CATEGORIES,
        "Current Stock": stock,
        "Projected 30-Day Demand": demand_30d,
        "Safety Stock Level": safety_stock,
        "Restock Needed": restock_needed,
        "Days Until Stockout": np.where(demand_30d > 0,
                                         np.round(stock / (demand_30d / 30), 1),
                                         999),
    })


def get_forecast_metrics():
    """Forecast model evaluation metrics per category."""
    np.random.seed(33)
    return pd.DataFrame({
        "Category": CATEGORIES,
        "MAPE (%)": np.round(np.random.uniform(4, 18, len(CATEGORIES)), 1),
        "MAE": np.round(np.random.uniform(8, 35, len(CATEGORIES)), 1),
        "Coverage (%)": np.round(np.random.uniform(88, 97, len(CATEGORIES)), 1),
    })


def get_product_categories():
    """List of available product categories."""
    return CATEGORIES
