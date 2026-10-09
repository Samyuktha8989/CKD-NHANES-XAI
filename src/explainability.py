"""
SHAP explainability, feature aggregation, and local explanation extraction module.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Union
import shap

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)


def compute_fold_shap_values(
    model: Any,
    model_name: str,
    X_train_transformed: np.ndarray,
    X_test_transformed: np.ndarray,
    feature_names: List[str]
) -> np.ndarray:
    """
    Compute SHAP values for held-out validation observations using appropriate model explainer.
    Returns shap_values array of shape (N_test, N_features) for the positive CKD class.
    """
    model_name_lower = model_name.lower()

    if "random_forest" in model_name_lower or "random forest" in model_name_lower:
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_test_transformed)
        # In multi-class output (e.g. [class0_shap, class1_shap])
        if isinstance(raw_shap, list):
            shap_vals = raw_shap[1]  # positive class
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_vals = raw_shap[:, :, 1]
        else:
            shap_vals = raw_shap

    elif "xgboost" in model_name_lower:
        # Native TreeSHAP computation embedded in XGBoost C++ core (Lundberg et al.)
        import xgboost as xgb
        booster = model.get_booster()
        dtest = xgb.DMatrix(X_test_transformed, feature_names=feature_names)
        contribs = booster.predict(dtest, pred_contribs=True)
        # Last column is base bias term; columns 0..N-1 are feature SHAP attributions
        shap_vals = contribs[:, :-1]

    elif "logistic" in model_name_lower:
        # Linear explainer using background sample
        bg_sample = shap.sample(X_train_transformed, min(100, len(X_train_transformed)), random_state=42)
        explainer = shap.LinearExplainer(model, bg_sample)
        shap_vals = explainer.shap_values(X_test_transformed)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    else:
        # Fallback to KernelExplainer
        bg_sample = shap.sample(X_train_transformed, min(50, len(X_train_transformed)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, bg_sample)
        raw_shap = explainer.shap_values(X_test_transformed, nsamples=100)
        shap_vals = raw_shap[1] if isinstance(raw_shap, list) else raw_shap

    return np.asarray(shap_vals)


def aggregate_shap_to_parent_features(
    shap_matrix: np.ndarray,
    transformed_feature_names: List[str],
    column_to_original_map: Dict[str, str],
    original_feature_order: List[str]
) -> pd.DataFrame:
    """
    Aggregate transformed/one-hot feature SHAP attributions back to original clinical features.
    For each observation i and parent feature F, attribution is sum of absolute SHAP values across its dummy levels:
    attr_i(F) = sum_{k} |shap_{i, k}|.
    Returns DataFrame of shape (N_samples, N_original_features).
    """
    N_samples = shap_matrix.shape[0]
    agg_dict = {feat: np.zeros(N_samples) for feat in original_feature_order}

    for idx, col_name in enumerate(transformed_feature_names):
        parent_feat = column_to_original_map.get(col_name)
        if parent_feat in agg_dict:
            # Add absolute attribution contribution
            agg_dict[parent_feat] += np.abs(shap_matrix[:, idx])

    agg_df = pd.DataFrame(agg_dict, columns=original_feature_order)
    return agg_df


def compute_global_feature_importance(
    aggregated_shap_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP importance per original feature and rank features.
    """
    mean_abs_importance = aggregated_shap_df.mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": mean_abs_importance.index,
        "Mean_Abs_SHAP": mean_abs_importance.values
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    importance_df["Rank"] = np.arange(1, len(importance_df) + 1)
    return importance_df


def select_representative_local_cases(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, int]:
    """
    Reproducibly select indices for True Positive, True Negative, False Positive, False Negative cases.
    Picks the most confident case in each category.
    """
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)

    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]

    selected = {}
    if len(tp_indices) > 0:
        # Highest probability TP
        selected["True_Positive"] = int(tp_indices[np.argmax(y_prob[tp_indices])])
    if len(tn_indices) > 0:
        # Lowest probability TN
        selected["True_Negative"] = int(tn_indices[np.argmin(y_prob[tn_indices])])
    if len(fp_indices) > 0:
        # Highest probability FP
        selected["False_Positive"] = int(fp_indices[np.argmax(y_prob[fp_indices])])
    if len(fn_indices) > 0:
        # Lowest probability FN
        selected["False_Negative"] = int(fn_indices[np.argmin(y_prob[fn_indices])])

    return selected
