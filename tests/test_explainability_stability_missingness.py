"""
Unit tests for SHAP aggregation, stability metrics, and missingness masking.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.explainability import aggregate_shap_to_parent_features, compute_global_feature_importance, select_representative_local_cases
from src.stability_analysis import compute_pairwise_stability_metrics, compute_feature_rank_variability, compute_cross_model_agreement
from src.missingness_sensitivity import apply_mcar_masking


def test_aggregate_shap_to_parent_features():
    # 2 samples, 3 transformed features (num1, cat1_A, cat1_B)
    shap_mat = np.array([
        [0.5, 0.2, -0.1],
        [0.1, -0.4, 0.3]
    ])
    tf_names = ["num__num1", "cat__cat1_A", "cat__cat1_B"]
    col_map = {"num__num1": "num1", "cat__cat1_A": "cat1", "cat__cat1_B": "cat1"}
    orig_order = ["num1", "cat1"]

    agg_df = aggregate_shap_to_parent_features(shap_mat, tf_names, col_map, orig_order)
    assert agg_df.shape == (2, 2)
    # Sample 0: num1 = 0.5, cat1 = |0.2| + |-0.1| = 0.3
    assert np.isclose(agg_df.loc[0, "num1"], 0.5)
    assert np.isclose(agg_df.loc[0, "cat1"], 0.3)
    # Sample 1: num1 = 0.1, cat1 = |-0.4| + |0.3| = 0.7
    assert np.isclose(agg_df.loc[1, "num1"], 0.1)
    assert np.isclose(agg_df.loc[1, "cat1"], 0.7)


def test_stability_calculations():
    df_fold1 = pd.DataFrame({"Feature": ["A", "B", "C"], "Rank": [1, 2, 3], "Mean_Abs_SHAP": [0.5, 0.3, 0.1]})
    df_fold2 = pd.DataFrame({"Feature": ["A", "B", "C"], "Rank": [1, 3, 2], "Mean_Abs_SHAP": [0.6, 0.2, 0.3]})

    pairwise_df = compute_pairwise_stability_metrics([df_fold1, df_fold2], top_k_list=[2])
    assert len(pairwise_df) == 1
    assert pairwise_df["Spearman_Rank_Correlation"].iloc[0] == 0.5
    # Top 2 features: fold1: {A, B}, fold2: {A, C} -> intersection 1, union 3 -> Jaccard = 1/3
    assert np.isclose(pairwise_df["Top_2_Jaccard"].iloc[0], 1/3)


def test_apply_mcar_masking():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    })
    total_cells = 20
    masked_df = apply_mcar_masking(df, mask_fraction=0.20, random_state=42)
    n_masked = masked_df.isnull().sum().sum()
    assert n_masked == 4  # 20 * 0.20 = 4 cells masked
