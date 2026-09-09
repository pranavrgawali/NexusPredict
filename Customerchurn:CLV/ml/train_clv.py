import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Import from your modular files
from customer_features import build_clv_features
from evaluate import evaluate_clv
from mock_data import get_mock_data

def train_and_save_clv_model():
    print("Loading mock data...")
    users_df, transactions_df = get_mock_data()

    X, y, clv_features = build_clv_features(users_df, transactions_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_mae, rf_rmse = evaluate_clv(y_test, rf_model.predict(X_test))
    print(f"Random Forest -> MAE: {rf_mae:.2f} | RMSE: {rf_rmse:.2f}")

    # 2. Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_mae, gb_rmse = evaluate_clv(y_test, gb_model.predict(X_test))
    print(f"Gradient Boosting -> MAE: {gb_mae:.2f} | RMSE: {gb_rmse:.2f}")

    # 3. Model Selection based on lowest RMSE
    if rf_rmse < gb_rmse:
        best_model, best_name, best_mae, best_rmse = rf_model, "RandomForestRegressor", rf_mae, rf_rmse
    else:
        best_model, best_name, best_mae, best_rmse = gb_model, "GradientBoostingRegressor", gb_mae, gb_rmse

    print(f"Selected {best_name} based on lower RMSE.")

    # --- NEW FOLDER SAVING LOGIC ---
    # Get the directory of this script (ml folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and define the models folder path
    models_dir = os.path.join(script_dir, "..", "models")
    # Automatically create the models folder if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Define the exact file path
    save_path = os.path.join(models_dir, "clv_model.joblib")

    # Serialize Artifact Contract
    joblib.dump({
        "model": best_model,
        "features": clv_features,
        "model_name": best_name,
        "mae": best_mae,
        "rmse": best_rmse
    }, save_path)
    print(f"Saved model to {save_path} successfully.")

if __name__ == "__main__":
    train_and_save_clv_model()