"""
Unit tests for data loading, schema validation, and cohort filtering.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure deps and src are discoverable
deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.data_loading import load_config, load_raw_dataset, filter_valid_cohort, prepare_feature_target_split


def test_load_config():
    config = load_config("config/project.yaml")
    assert "data" in config
    assert "features" in config
    assert "validation" in config
    assert "models" in config
    assert config["validation"]["n_splits"] == 5
    assert config["validation"]["n_repeats"] == 5


def test_load_raw_dataset():
    raw_df = load_raw_dataset("CKD_NHANES_2021_2023.csv")
    assert raw_df.shape == (11933, 29)
    assert "participant_id" in raw_df.columns
    assert "ckd_present" in raw_df.columns
    assert "ckd_stage" in raw_df.columns


def test_filter_valid_cohort():
    raw_df = load_raw_dataset("CKD_NHANES_2021_2023.csv")
    valid_df, audit_info = filter_valid_cohort(raw_df)
    assert len(valid_df) == 6326
    assert audit_info["filtered_unknown_records"] == 5607
    assert (valid_df["ckd_stage"] == "Unknown").sum() == 0
    # Target distribution
    counts = valid_df["ckd_present"].value_counts()
    assert counts[0] == 3592
    assert counts[1] == 2734


def test_prepare_feature_target_split_leakage_safety():
    config = load_config("config/project.yaml")
    raw_df = load_raw_dataset("CKD_NHANES_2021_2023.csv")
    valid_df, _ = filter_valid_cohort(raw_df)
    X, y, num_feats, cat_feats = prepare_feature_target_split(valid_df, config)

    assert len(X) == 6326
    assert len(y) == 6326
    assert X.shape[1] == 21

    # Verify no leakage columns exist in X
    excluded = ["egfr", "serum_creatinine", "urine_albumin", "urine_creatinine", "albumin_creatinine_ratio", "ckd_stage", "participant_id"]
    for col in excluded:
        assert col not in X.columns, f"Leakage column {col} was found in X!"
