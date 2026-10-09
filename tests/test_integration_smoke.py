"""
Integration and smoke test on isolated synthetic toy data fixture.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import shutil

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.models import get_all_models
from src.validation import run_repeated_cv
from src.metrics import aggregate_cv_metrics


def test_smoke_integration_pipeline(tmp_path):
    # Create small synthetic test dataset
    np.random.seed(42)
    n_samples = 40
    df = pd.DataFrame({
        "num1": np.random.randn(n_samples),
        "num2": np.random.rand(n_samples) * 50,
        "cat1": np.random.choice(["TypeA", "TypeB"], size=n_samples),
        "target": np.random.choice([0, 1], size=n_samples)
    })

    test_config = {
        "validation": {
            "n_splits": 2,
            "n_repeats": 2,
            "random_state": 42,
            "decision_threshold": 0.5
        },
        "explainability": {
            "shap_sample_limit": 20
        },
        "models": {
            "logistic_regression": {"enabled": True, "params": {"max_iter": 100}},
            "random_forest": {"enabled": True, "params": {"n_estimators": 5, "min_samples_leaf": 2}},
            "xgboost": {"enabled": True, "params": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1}}
        }
    }

    models = get_all_models(test_config, seed=42)
    X = df[["num1", "num2", "cat1"]]
    y = df["target"]

    out_dirs = {
        "metrics_dir": str(tmp_path / "metrics"),
        "predictions_dir": str(tmp_path / "predictions")
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    cv_res = run_repeated_cv(
        X=X,
        y=y,
        numeric_features=["num1", "num2"],
        categorical_features=["cat1"],
        models=models,
        config=test_config,
        output_dirs=out_dirs
    )

    assert len(cv_res["fold_metrics"]) == 3 * 4  # 3 models * 4 folds
    assert os.path.exists(os.path.join(out_dirs["metrics_dir"], "fold_metrics.csv"))
    assert os.path.exists(os.path.join(out_dirs["predictions_dir"], "validation_predictions.csv"))

    agg = aggregate_cv_metrics(cv_res["fold_metrics"])
    assert len(agg) == 3
