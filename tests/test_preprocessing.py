"""
Unit tests for leakage-safe preprocessing and feature name mapping.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.preprocessing import build_preprocessor, get_feature_names_and_mapping


def test_build_preprocessor_and_transform():
    # Synthetic test data
    df_train = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0],
        "num2": [10.0, 20.0, 30.0, np.nan],
        "cat1": ["A", "B", "A", np.nan],
        "cat2": [1.0, 2.0, np.nan, 1.0]
    })
    df_test = pd.DataFrame({
        "num1": [np.nan, 5.0],
        "num2": [50.0, np.nan],
        "cat1": ["C", "A"],  # "C" is unseen category
        "cat2": [np.nan, 2.0]
    })

    num_cols = ["num1", "num2"]
    cat_cols = ["cat1", "cat2"]

    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    X_train_tf = preprocessor.fit_transform(df_train)
    X_test_tf = preprocessor.transform(df_test)

    assert not np.isnan(X_train_tf).any()
    assert not np.isnan(X_test_tf).any()
    assert X_train_tf.shape[0] == 4
    assert X_test_tf.shape[0] == 2

    # Verify feature name mapping
    tf_names, col_map = get_feature_names_and_mapping(preprocessor, num_cols, cat_cols)
    assert len(tf_names) == X_train_tf.shape[1]
    for tf_name in tf_names:
        assert tf_name in col_map
        orig = col_map[tf_name]
        assert orig in num_cols + cat_cols
