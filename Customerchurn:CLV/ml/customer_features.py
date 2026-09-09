import pandas as pd

def build_churn_features(users_df, transactions_df):
    """Engineers features and target for the Churn Classification model."""
    tx_agg = transactions_df.groupby("user_id").agg(
        transaction_count=('transaction_id', 'count'),
        total_spend=('amount_spent', 'sum'),
        average_order_value=('amount_spent', 'mean'),
        last_purchase_date=('purchase_date', 'max')
    ).reset_index()

    df = pd.merge(users_df, tx_agg, on="user_id", how="left")
    df['transaction_count'] = df['transaction_count'].fillna(0)
    df['total_spend'] = df['total_spend'].fillna(0.0)
    df['average_order_value'] = df['average_order_value'].fillna(0.0)

    # Feature Engineering: Recency days
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])
    reference_date = df["last_purchase_date"].max()
    df["recency_days"] = (reference_date - df["last_purchase_date"]).dt.days
    df["recency_days"] = df["recency_days"].fillna(999)

    # Churn Target: Y = 1 when recency > 30 and login frequency == 0
    df["churn"] = ((df["recency_days"] > 30) & (df["login_frequency_per_month"] == 0)).astype(int)

    features = [
        "age", "login_frequency_per_month", "support_tickets_raised",
        "is_active_subscriber", "transaction_count", "total_spend",
        "average_order_value", "recency_days"
    ]
    return df[features], df["churn"], features


def build_clv_features(users_df, transactions_df):
    """Engineers historical RFM features and the future 6-month target for CLV regression."""
    transactions_df['purchase_date'] = pd.to_datetime(transactions_df['purchase_date'])
    
    # Split historical observation window vs. future 6-month target window
    max_date = transactions_df['purchase_date'].max()
    cutoff_date = max_date - pd.DateOffset(months=6)

    history_tx = transactions_df[transactions_df['purchase_date'] <= cutoff_date]
    future_tx = transactions_df[transactions_df['purchase_date'] > cutoff_date]

    hist_agg = history_tx.groupby("user_id").agg(
        transaction_count=('transaction_id', 'count'),
        total_spend=('amount_spent', 'sum'),
        average_order_value=('amount_spent', 'mean'),
        last_purchase_date=('purchase_date', 'max')
    ).reset_index()

    # Future target spend (6-month CLV)
    future_clv = future_tx.groupby("user_id")["amount_spent"].sum().reset_index()
    future_clv.rename(columns={'amount_spent': 'future_6_month_clv'}, inplace=True)

    df = pd.merge(users_df, hist_agg, on="user_id", how="left")
    df = pd.merge(df, future_clv, on="user_id", how="left")

    df['transaction_count'] = df['transaction_count'].fillna(0)
    df['total_spend'] = df['total_spend'].fillna(0.0)
    df['average_order_value'] = df['average_order_value'].fillna(0.0)
    df['future_6_month_clv'] = df['future_6_month_clv'].fillna(0.0)

    reference_date = history_tx['purchase_date'].max()
    df["recency_days"] = (reference_date - pd.to_datetime(df["last_purchase_date"])).dt.days
    df["recency_days"] = df["recency_days"].fillna(999)

    clv_features = [
        "recency_days", "transaction_count", "total_spend", "average_order_value",
        "login_frequency_per_month", "support_tickets_raised", "is_active_subscriber"
    ]
    return df[clv_features], df["future_6_month_clv"], clv_features