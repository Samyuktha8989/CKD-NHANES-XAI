"""
Model instantiation and configuration module for CKD classification.
"""

import os
import sys
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)


def get_model(model_name: str, config: Dict[str, Any], seed: int = None) -> Any:
    """
    Instantiate model by name with hyperparameter settings from config.
    """
    model_name_lower = model_name.lower().replace(" ", "_")

    if model_name_lower in ["logistic_regression", "lr"]:
        params = config["models"]["logistic_regression"]["params"].copy()
        if seed is not None:
            params["random_state"] = seed
        return LogisticRegression(**params)

    elif model_name_lower in ["random_forest", "rf"]:
        params = config["models"]["random_forest"]["params"].copy()
        if seed is not None:
            params["random_state"] = seed
        return RandomForestClassifier(**params)

    elif model_name_lower in ["xgboost", "xgb"]:
        params = config["models"]["xgboost"]["params"].copy()
        if seed is not None:
            params["random_state"] = seed
        return XGBClassifier(**params)

    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose from: 'logistic_regression', 'random_forest', 'xgboost'")


def get_all_models(config: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
    """
    Get dictionary of all enabled models.
    """
    models = {}
    if config["models"]["logistic_regression"].get("enabled", True):
        models["Logistic Regression"] = get_model("logistic_regression", config, seed)
    if config["models"]["random_forest"].get("enabled", True):
        models["Random Forest"] = get_model("random_forest", config, seed)
    if config["models"]["xgboost"].get("enabled", True):
        models["XGBoost"] = get_model("xgboost", config, seed)
    return models
