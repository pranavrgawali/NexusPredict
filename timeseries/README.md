# Role 3 -> Role 4 Integration Handoff

Hello! This folder contains everything you need from **Role 3 (Supply Chain & Time-Series Forecasting)** to build the Supply Chain Dashboard in Streamlit. 

The Prophet models are **already trained and serialized**. As per the project requirements, **do not train models inside Streamlit**. You just need to load them and generate forecasts using the utilities provided.

## 📂 Folder Structure

You can drop these folders directly into your project root:

```text
/
├── models/
│   └── prophet/                 # Contains 8 pre-trained .joblib model artifacts
│       ├── Electronics.joblib
│       ├── Clothing.joblib
│       └── ...
└── utils/
    ├── model_loader.py          # Functions to load the serialized models
    └── forecast_utils.py        # Functions to generate 30-day forecasts
```

## 🚀 How to Integrate with Streamlit

You only need to interact with the two files inside the `utils/` folder. Here is the exact code you need to render the Supply Chain page.

### 1. Setup and Imports

At the top of your Streamlit page (e.g., `pages/supply_chain.py`), import the functions:

```python
import streamlit as st
from utils.model_loader import load_prophet_model, get_available_categories
from utils.forecast_utils import generate_forecast, calculate_stock_gap
```

### 2. Loading the Model (The Dropdown)

Use `get_available_categories()` to automatically populate your category dropdown, then use `load_prophet_model()` to load the artifact into memory.

```python
st.title("Supply Chain & Inventory Forecasting")

# 1. Get categories and create dropdown
categories = get_available_categories()
selected_category = st.selectbox("Select Product Category", categories)

# 2. Load the pre-trained artifact for the selected category
try:
    bundle = load_prophet_model(selected_category)
    model = bundle["model"]
    mape_score = bundle["mape"]
    
    st.info(f"Model loaded successfully! (Historical Error Rate: {mape_score}%)")
except FileNotFoundError:
    st.error("Model artifacts not found. Make sure models/prophet is in the root directory.")
    st.stop()
```

### 3. Generating the Forecast & Stock Gap

Pass the model to `generate_forecast(model)` to get the next 30 days of predictions. Then use `calculate_stock_gap` to figure out if you need to order more stock.

```python
# 3. Generate 30-day forecast
# Returns a dataframe with columns: ds (date), yhat (demand), yhat_lower, yhat_upper
forecast_df = generate_forecast(model, periods=30)

# 4. Fetch the CURRENT stock from YOUR database
# (Replace this with your actual DB query to Role 1's table)
# Example: 
# current_stock = db.query(f"SELECT stock_level_available FROM daily_inventory_sales WHERE product_category = '{selected_category}' ORDER BY log_date DESC LIMIT 1")
current_stock = 250 # Placeholder integer

# 5. Calculate stock gap
gap_metrics = calculate_stock_gap(forecast_df, current_stock)

# Display Metrics
col1, col2, col3 = st.columns(3)
col1.metric("30-Day Forecast Demand", f"{gap_metrics['forecast_units']:,} units")
col2.metric("Current Stock Level", f"{gap_metrics['current_stock']:,} units")

# If recommended order is > 0, show it in red/warning
if gap_metrics['recommended_order'] > 0:
    col3.metric("Recommended Order", f"{gap_metrics['recommended_order']:,} units", delta="-Deficit", delta_color="inverse")
else:
    col3.metric("Surplus Stock", f"{gap_metrics['surplus_or_deficit']:,} units", delta="+Surplus")
```

### 4. Plotting the Chart

You can pass `forecast_df` directly into Streamlit's line chart, or use Plotly for a better look.

```python
# Streamlit native chart
st.subheader("30-Day Demand Projection")
chart_data = forecast_df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
chart_data.columns = ["Predicted Demand", "Pessimistic Bound", "Optimistic Bound"]

st.line_chart(chart_data)
```

## ⚠️ Important Rules for Role 4
1. **Never call `.fit()`**. The models are strictly pre-trained.
2. **Negative values are handled**. `generate_forecast()` automatically clips any negative demand predictions to `0`, so your charts won't drop below the X-axis.
3. **Database connection**: You will connect to the `daily_inventory_sales` table independently to get the `current_stock` value needed for step 4. Our forecasting models do not need live database access during inference.

Reach out to Role 3 if you run into any issues loading the models!
