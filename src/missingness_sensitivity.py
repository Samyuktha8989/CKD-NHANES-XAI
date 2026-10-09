"""
Controlled missing-data sensitivity and stress-testing module.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy.stats import spearmanr
from src.metrics import calculate_binary_metrics
from src.explainability import (
    compute_fold_shap_values,
    aggregate_shap_to_parent_features,
    compute_global_feature_importance
)

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)


def apply_mcar_masking(
    X: pd.DataFrame,
    mask_fraction: float,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Apply Missing Completely at Random (MCAR) masking to observed predictor cells.
    Does not mask target or identifier columns.
    """
    if mask_fraction <= 0.0:
        return X.copy()

    rng = np.random.RandomState(random_state)
    X_masked = X.copy()

    # Find non-null cell coordinates
    non_null_coords = []
    for row_idx in range(len(X)):
        for col_idx, col_name in enumerate(X.columns):
            if pd.notnull(X.iat[row_idx, col_idx]):
                non_null_coords.append((row_idx, col_idx))

    n_to_mask = int(np.round(len(non_null_coords) * mask_fraction))
    if n_to_mask > 0 and non_null_coords:
        chosen_indices = rng.choice(len(non_null_coords), size=n_to_mask, replace=False)
        for idx in chosen_indices:
            r, c = non_null_coords[idx]
            X_masked.iat[r, c] = np.nan

    return X_masked


def evaluate_missingness_stress_test(
    model: Any,
    model_name: str,
    fitted_preprocessor: Any,
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
    baseline_importance_df: pd.DataFrame,
    transformed_feature_names: List[str],
    column_to_original_map: Dict[str, str],
    original_feature_order: List[str],
    masking_levels: List[float] = [0.10, 0.20],
    masking_seeds: List[int] = [42, 101, 202, 303, 404],
    compute_shap: bool = True,
    shap_sample_limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Evaluate trained model against masked validation inputs without refitting preprocessor or model.
    """
    results = []

    # Baseline (0% additional missingness)
    X_test_base_tf = fitted_preprocessor.transform(X_test_raw)
    y_prob_base = model.predict_proba(X_test_base_tf)[:, 1]
    base_metrics = calculate_binary_metrics(y_test, y_prob_base)

    base_record = {
        "Model": model_name,
        "Masking_Level": 0.0,
        "Seed": 0,
        "ROC_AUC": base_metrics["ROC_AUC"],
        "PR_AUC": base_metrics["PR_AUC"],
        "Sensitivity": base_metrics["Sensitivity"],
        "Specificity": base_metrics["Specificity"],
        "F1_Score": base_metrics["F1_Score"],
        "Accuracy": base_metrics["Accuracy"],
        "Brier_Score": base_metrics["Brier_Score"],
        "SHAP_Rank_Correlation_With_Baseline": 1.0
    }
    results.append(base_record)

    # Stress testing across levels and seeds
    for level in masking_levels:
        for seed in masking_seeds:
            X_masked = apply_mcar_masking(X_test_raw, mask_fraction=level, random_state=seed)
            # Transform using ALREADY FITTED preprocessor
            X_masked_tf = fitted_preprocessor.transform(X_masked)
            y_prob_masked = model.predict_proba(X_masked_tf)[:, 1]
            masked_metrics = calculate_binary_metrics(y_test, y_prob_masked)

            # Measure explanation stability if enabled
            rank_corr = np.nan
            if compute_shap:
                try:
                    shap_eval_tf = X_masked_tf[:shap_sample_limit] if (shap_sample_limit and len(X_masked_tf) > shap_sample_limit) else X_masked_tf
                    shap_mat = compute_fold_shap_values(
                        model=model,
                        model_name=model_name,
                        X_train_transformed=X_masked_tf[:50],  # minimal bg
                        X_test_transformed=shap_eval_tf,
                        feature_names=transformed_feature_names
                    )
                    agg_shap_df = aggregate_shap_to_parent_features(
                        shap_matrix=shap_mat,
                        transformed_feature_names=transformed_feature_names,
                        column_to_original_map=column_to_original_map,
                        original_feature_order=original_feature_order
                    )
                    masked_imp_df = compute_global_feature_importance(agg_shap_df).set_index("Feature")
                    base_imp_indexed = baseline_importance_df.set_index("Feature")
                    common_feats = base_imp_indexed.index.intersection(masked_imp_df.index)
                    corr_val, _ = spearmanr(base_imp_indexed.loc[common_feats, "Rank"], masked_imp_df.loc[common_feats, "Rank"])
                    rank_corr = float(corr_val)
                except Exception:
                    rank_corr = np.nan

            record = {
                "Model": model_name,
                "Masking_Level": level,
                "Seed": seed,
                "ROC_AUC": masked_metrics["ROC_AUC"],
                "PR_AUC": masked_metrics["PR_AUC"],
                "Sensitivity": masked_metrics["Sensitivity"],
                "Specificity": masked_metrics["Specificity"],
                "F1_Score": masked_metrics["F1_Score"],
                "Accuracy": masked_metrics["Accuracy"],
                "Brier_Score": masked_metrics["Brier_Score"],
                "SHAP_Rank_Correlation_With_Baseline": rank_corr
            }
            results.append(record)

    return results
