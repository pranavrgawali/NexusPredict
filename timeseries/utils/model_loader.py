import os
import joblib
import logging
import json

logger = logging.getLogger(__name__)

def _get_default_model_dir():
    try:
        from ml.forecast_config import MODEL_OUTPUT_DIR
        return MODEL_OUTPUT_DIR
    except ImportError:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "models", "prophet")

def load_prophet_model(category, model_dir=None):
    """
    Load a serialized Prophet model artifact for a specific product category.

    Args:
        category (str): Product category name (e.g., "Electronics")
        model_dir (str): Directory containing .joblib files. Defaults to models/prophet/

    Returns:
        dict: {"model": Prophet, "category": str, "mape": float}

    Raises:
        FileNotFoundError: If the artifact file does not exist
        KeyError: If the artifact is missing required keys
    """
    model_dir = model_dir or _get_default_model_dir()
    filepath = os.path.join(model_dir, f"{category}.joblib")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Prophet artifact not found: {filepath}. "
            f"Run 'python -m ml.train_forecast' to train models first."
        )

    bundle = joblib.load(filepath)

    # Validate contract
    required_keys = {"model", "category", "mape"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise KeyError(f"Artifact missing required keys: {missing}")

    logger.info(f"Loaded Prophet model for '{category}' (MAPE: {bundle['mape']}%)")
    return bundle


def get_available_categories(model_dir=None):
    """
    List all product categories with trained Prophet models.

    Returns:
        list[str]: Category names (derived from .joblib filenames)
    """
    model_dir = model_dir or _get_default_model_dir()
    if not os.path.exists(model_dir):
        return []

    categories = [
        f.replace(".joblib", "")
        for f in os.listdir(model_dir)
        if f.endswith(".joblib") and f != "evaluation_summary.json"
    ]
    return sorted(categories)


def load_evaluation_summary(model_dir=None):
    """
    Load the evaluation summary JSON for all categories.

    Returns:
        dict: Full evaluation results or empty dict if not found
    """
    model_dir = model_dir or _get_default_model_dir()
    filepath = os.path.join(model_dir, "evaluation_summary.json")

    if not os.path.exists(filepath):
        return {}

    with open(filepath, "r", encoding='utf-8') as f:
        return json.load(f)
