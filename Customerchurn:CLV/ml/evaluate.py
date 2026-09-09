import numpy as np
from sklearn.metrics import recall_score, roc_auc_score, mean_absolute_error, mean_squared_error

def evaluate_churn(y_true, y_pred, y_prob):
    """Calculates Recall and ROC-AUC for classification."""
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    return recall, roc_auc

def evaluate_clv(y_true, y_pred):
    """Calculates MAE and RMSE for regression."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse