"""
Data loading and cohort definition module for CKD-NHANES 2021-2023 dataset.
"""

import os
import sys
import yaml
import pandas as pd
from typing import Dict, Any, Tuple, List

# Add deps if needed
deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)


def load_config(config_path: str = "config/project.yaml") -> Dict[str, Any]:
    """Load YAML project configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_raw_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the raw CSV dataset from disk."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    df = pd.read_csv(dataset_path)
    return df


def filter_valid_cohort(
    df: pd.DataFrame,
    stage_col: str = "ckd_stage",
    invalid_stages: List[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filter dataset to retain participants with verifiable CKD staging.
    Removes participants with 'Unknown' stage (untested eGFR / pediatric sample).
    """
    if invalid_stages is None:
        invalid_stages = ["Unknown"]

    raw_count = len(df)
    valid_mask = ~df[stage_col].isin(invalid_stages)
    valid_df = df[valid_mask].copy().reset_index(drop=True)
    filtered_count = raw_count - len(valid_df)

    audit_info = {
        "raw_total_records": raw_count,
        "filtered_unknown_records": filtered_count,
        "valid_cohort_records": len(valid_df),
        "target_distribution_valid": valid_df["ckd_present"].value_counts().to_dict(),
        "stage_distribution_valid": valid_df[stage_col].value_counts().to_dict()
    }
    return valid_df, audit_info


def prepare_feature_target_split(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Extract approved predictor matrix X and binary target vector y based on config exclusions.
    """
    target_col = config["data"]["target_column"]
    excluded_ids = config["features"].get("excluded_identifiers", [])
    excluded_leakage = config["features"].get("excluded_leakage", [])
    all_excluded = set(excluded_ids + excluded_leakage + [target_col])

    numeric_features = config["features"]["numeric_features"]
    categorical_features = config["features"]["categorical_features"]
    approved_features = numeric_features + categorical_features

    # Verify approved features do not overlap with excluded features
    overlap = set(approved_features).intersection(all_excluded)
    if overlap:
        raise ValueError(f"Fatal leakage risk: approved features contain excluded variables: {overlap}")

    # Check available columns in df
    missing_cols = [col for col in approved_features if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Expected predictor columns missing from dataset: {missing_cols}")

    X = df[approved_features].copy()
    y = df[target_col].astype(int).copy()

    return X, y, numeric_features, categorical_features
