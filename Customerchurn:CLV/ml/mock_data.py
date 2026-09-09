import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_mock_data():
    # Set your mock data sizes here! 
    # (Set to 100 for instant testing, or 10000 for realistic testing)
    N_USERS = 100
    N_TRANS = 500
    
    # 1. Mock Users Table
    users = pd.DataFrame({
        "user_id": range(1, N_USERS + 1),
        "signup_date": pd.date_range(start="2024-01-01", periods=N_USERS),
        "age": np.random.randint(18, 65, size=N_USERS),
        "country": ["USA"] * N_USERS,
        "login_frequency_per_month": np.random.randint(0, 20, size=N_USERS),
        "support_tickets_raised": np.random.randint(0, 5, size=N_USERS),
        "is_active_subscriber": np.random.choice([0, 1], size=N_USERS)
    })

    # 2. Mock Transactions Table
    # Create random dates over the last 8 months (240 days)
    dates = [datetime.today() - timedelta(days=int(x)) for x in np.random.randint(0, 240, size=N_TRANS)]
    
    transactions = pd.DataFrame({
        "transaction_id": range(1, N_TRANS + 1),
        "user_id": np.random.choice(range(1, N_USERS + 1), size=N_TRANS),
        "purchase_date": dates,
        "amount_spent": np.random.uniform(10.0, 500.0, size=N_TRANS),
        "items_count": np.random.randint(1, 5, size=N_TRANS)
    })

    return users, transactions