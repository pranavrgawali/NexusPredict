import pandas as pd
import numpy as np

def generate_forecast(model, periods=30):
    """
    Generate a demand forecast for the specified number of days.

    Args:
        model: Fitted Prophet model (from load_prophet_model()["model"])
        periods: Number of future days to forecast (default: 30)

    Returns:
        pd.DataFrame with columns [ds, yhat, yhat_lower, yhat_upper]
    """
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)

    forecast_future = forecast[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].tail(periods).copy()

    # Demand cannot be negative
    forecast_future["yhat"] = forecast_future["yhat"].clip(lower=0)
    forecast_future["yhat_lower"] = forecast_future["yhat_lower"].clip(lower=0)
    forecast_future["yhat_upper"] = forecast_future["yhat_upper"].clip(lower=0)

    return forecast_future

def calculate_stock_gap(forecast_df, current_stock):
    """
    Compare forecasted demand with current stock levels.

    Args:
        forecast_df: DataFrame from generate_forecast()
        current_stock: Current stock level (int)

    Returns:
        dict with forecast_units, current_stock, recommended_order, surplus_or_deficit
    """
    forecast_units = float(forecast_df["yhat"].sum())
    recommended_order = max(0.0, forecast_units - current_stock)

    return {
        "forecast_units": round(forecast_units),
        "current_stock": int(current_stock),
        "recommended_order": round(recommended_order),
        "surplus_or_deficit": round(current_stock - forecast_units)
    }

def format_forecast_for_chart(forecast_df):
    """
    Format forecast DataFrame for Streamlit chart rendering.

    Args:
        forecast_df: DataFrame from generate_forecast()

    Returns:
        pd.DataFrame with date index and renamed columns for display
    """
    chart_df = forecast_df.copy()
    chart_df = chart_df.set_index("ds")
    chart_df.columns = ["Predicted Demand", "Lower Bound", "Upper Bound"]
    return chart_df
