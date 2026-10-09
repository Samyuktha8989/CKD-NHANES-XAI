"""
Unit tests for binary performance metric calculations.
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

from src.metrics import calculate_binary_metrics, aggregate_cv_metrics


def test_calculate_binary_metrics_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])
    metrics = calculate_binary_metrics(y_true, y_prob)

    assert metrics["ROC_AUC"] == 1.0
    assert metrics["PR_AUC"] == 1.0
    assert metrics["Sensitivity"] == 1.0
    assert metrics["Specificity"] == 1.0
    assert metrics["Accuracy"] == 1.0
    assert metrics["F1_Score"] == 1.0
    assert metrics["MCC"] == 1.0
    assert metrics["TP"] == 2
    assert metrics["TN"] == 2
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0


def test_calculate_binary_metrics_imperfect():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.8, 0.7, 0.3, 0.2])  # 1 TP, 1 FP, 1 FN, 1 TN
    metrics = calculate_binary_metrics(y_true, y_prob)

    assert 0.0 <= metrics["ROC_AUC"] <= 1.0
    assert metrics["Sensitivity"] == 0.5
    assert metrics["Specificity"] == 0.5
    assert metrics["Accuracy"] == 0.5
    assert metrics["TP"] == 1
    assert metrics["FP"] == 1
    assert metrics["TN"] == 1
    assert metrics["FN"] == 1


def test_aggregate_cv_metrics():
    df_folds = pd.DataFrame([
        {"Model": "RF", "ROC_AUC": 0.80, "Accuracy": 0.75, "Sensitivity": 0.70},
        {"Model": "RF", "ROC_AUC": 0.84, "Accuracy": 0.77, "Sensitivity": 0.72}
    ])
    agg = aggregate_cv_metrics(df_folds)
    assert len(agg) == 1
    assert agg["ROC_AUC_Mean"].iloc[0] == 0.82
    assert "0.820 ± 0.028" in agg["ROC_AUC_Summary"].iloc[0]
