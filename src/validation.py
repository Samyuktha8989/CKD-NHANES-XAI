"""
Cross-validation engine and fold-level evaluation orchestrator.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib

from src.preprocessing import build_preprocessor, get_feature_names_and_mapping
from src.metrics import calculate_binary_metrics
from src.explainability import (
    compute_fold_shap_values,
    aggregate_shap_to_parent_features,
    compute_global_feature_importance
)

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)


def run_repeated_cv(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    models: Dict[str, Any],
    config: Dict[str, Any],
    output_dirs: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Execute Repeated Stratified 5-Fold Cross-Validation across all models.
    Guarantees zero leakage: all preprocessors fitted strictly on training observations.
    """
    n_splits = config["validation"]["n_splits"]
    n_repeats = config["validation"]["n_repeats"]
    seed = config["validation"]["random_state"]
    threshold = config["validation"].get("decision_threshold", 0.5)
    shap_limit = config["explainability"].get("shap_sample_limit", None)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    all_fold_metrics = []
    all_predictions = []
    fold_importance_by_model = {m: [] for m in models.keys()}
    all_fold_shap_by_model = {m: [] for m in models.keys()}
    fitted_pipelines_first_fold = {}

    total_folds = n_splits * n_repeats
    print(f"Starting {total_folds}-Fold Repeated Stratified Cross-Validation for {len(models)} models...")

    fold_idx = 0
    for train_idx, val_idx in rskf.split(X, y):
        X_train_raw, X_val_raw = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

        # Build and fit preprocessing strictly on X_train_raw
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        X_train_tf = preprocessor.fit_transform(X_train_raw)
        X_val_tf = preprocessor.transform(X_val_raw)

        tf_names, col_to_orig_map = get_feature_names_and_mapping(
            preprocessor, numeric_features, categorical_features
        )
        orig_features = numeric_features + categorical_features

        for model_name, base_model in models.items():
            # Clone model for this fold
            from sklearn.base import clone
            fold_model = clone(base_model)
            fold_model.fit(X_train_tf, y_train)

            # Save first fold pipeline for sensitivity testing or inspection
            if fold_idx == 0:
                fitted_pipelines_first_fold[model_name] = (fold_model, preprocessor, tf_names, col_to_orig_map)

            # Predict on untouched validation fold
            y_prob = fold_model.predict_proba(X_val_tf)[:, 1]
            metrics = calculate_binary_metrics(y_val, y_prob, threshold=threshold)
            metrics["Model"] = model_name
            metrics["Fold_Index"] = fold_idx
            metrics["Repeat_Index"] = fold_idx // n_splits
            metrics["Split_Index"] = fold_idx % n_splits
            metrics["Train_Samples"] = len(train_idx)
            metrics["Val_Samples"] = len(val_idx)
            all_fold_metrics.append(metrics)

            # Store predictions
            for v_idx, true_lbl, prob_val in zip(val_idx, y_val, y_prob):
                all_predictions.append({
                    "Model": model_name,
                    "Fold_Index": fold_idx,
                    "Row_Index": int(v_idx),
                    "y_true": int(true_lbl),
                    "y_prob": float(prob_val),
                    "y_pred": int(prob_val >= threshold)
                })

            # SHAP calculation for validation observations
            if shap_limit is not None and len(X_val_tf) > shap_limit:
                shap_eval_tf = X_val_tf[:shap_limit]
            else:
                shap_eval_tf = X_val_tf

            shap_matrix = compute_fold_shap_values(
                model=fold_model,
                model_name=model_name,
                X_train_transformed=X_train_tf,
                X_test_transformed=shap_eval_tf,
                feature_names=tf_names
            )

            # Aggregate to parent clinical features
            agg_shap_df = aggregate_shap_to_parent_features(
                shap_matrix=shap_matrix,
                transformed_feature_names=tf_names,
                column_to_original_map=col_to_orig_map,
                original_feature_order=orig_features
            )
            fold_importance_df = compute_global_feature_importance(agg_shap_df)
            fold_importance_df["Model"] = model_name
            fold_importance_df["Fold_Index"] = fold_idx
            fold_importance_by_model[model_name].append(fold_importance_df)

        fold_idx += 1
        if fold_idx % 5 == 0 or fold_idx == total_folds:
            print(f"  Completed fold {fold_idx} / {total_folds}")

    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    predictions_df = pd.DataFrame(all_predictions)

    # Save to disk if output_dirs provided
    if output_dirs:
        if "metrics_dir" in output_dirs:
            fold_metrics_df.to_csv(os.path.join(output_dirs["metrics_dir"], "fold_metrics.csv"), index=False)
        if "predictions_dir" in output_dirs:
            predictions_df.to_csv(os.path.join(output_dirs["predictions_dir"], "validation_predictions.csv"), index=False)

    return {
        "fold_metrics": fold_metrics_df,
        "predictions": predictions_df,
        "fold_importance_by_model": fold_importance_by_model,
        "first_fold_pipelines": fitted_pipelines_first_fold
    }
