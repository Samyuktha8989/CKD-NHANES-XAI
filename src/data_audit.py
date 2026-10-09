"""
Dataset audit and quality analysis module.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)


def run_comprehensive_audit(
    raw_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    output_dir: str = "results/data_audit"
) -> Dict[str, pd.DataFrame]:
    """
    Run full data audit on raw and valid cohorts, saving tables to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset Dimensions & Overview
    overview_data = [
        {"Dataset Version": "Raw Kaggle NHANES 2021-2023", "Rows": raw_df.shape[0], "Columns": raw_df.shape[1], "Duplicates": raw_df.duplicated().sum()},
        {"Dataset Version": "Valid Analysed Cohort (KDIGO staged)", "Rows": valid_df.shape[0], "Columns": valid_df.shape[1], "Duplicates": valid_df.duplicated().sum()},
        {"Dataset Version": "Excluded (Unknown Stage / Unmeasured GFR)", "Rows": raw_df.shape[0] - valid_df.shape[0], "Columns": raw_df.shape[1], "Duplicates": 0}
    ]
    overview_df = pd.DataFrame(overview_data)
    overview_df.to_csv(os.path.join(output_dir, "dataset_overview.csv"), index=False)

    # 2. Missingness Audit (Raw vs Valid)
    missing_raw = raw_df.isnull().sum()
    missing_raw_pct = (missing_raw / len(raw_df)) * 100
    missing_valid = valid_df.isnull().sum()
    missing_valid_pct = (missing_valid / len(valid_df)) * 100

    missing_df = pd.DataFrame({
        "Column": raw_df.columns,
        "Dtype": raw_df.dtypes.astype(str),
        "Raw_Missing_Count": missing_raw.values,
        "Raw_Missing_Pct": missing_raw_pct.round(2).values,
        "Valid_Missing_Count": missing_valid.reindex(raw_df.columns).values,
        "Valid_Missing_Pct": missing_valid_pct.reindex(raw_df.columns).round(2).values
    })
    missing_df.to_csv(os.path.join(output_dir, "feature_missingness_audit.csv"), index=False)

    # 3. Target Distribution Audit
    target_counts = valid_df["ckd_present"].value_counts()
    target_pct = valid_df["ckd_present"].value_counts(normalize=True) * 100
    target_df = pd.DataFrame({
        "Target_Class": ["Class 0 (No CKD)", "Class 1 (CKD Present)"],
        "Count": [target_counts.get(0, 0), target_counts.get(1, 0)],
        "Percentage": [round(target_pct.get(0, 0.0), 2), round(target_pct.get(1, 0.0), 2)]
    })
    target_df.to_csv(os.path.join(output_dir, "target_distribution.csv"), index=False)

    # 4. Stage Cross-tabulation
    stage_counts = valid_df["ckd_stage"].value_counts().reset_index()
    stage_counts.columns = ["CKD_Stage", "Count"]
    stage_counts["Percentage"] = (stage_counts["Count"] / len(valid_df) * 100).round(2)
    stage_counts.to_csv(os.path.join(output_dir, "ckd_stage_breakdown.csv"), index=False)

    # 5. Descriptive Statistics of Approved Features in Valid Cohort
    desc_df = valid_df.describe(include="all").T.reset_index().rename(columns={"index": "Feature"})
    desc_df.to_csv(os.path.join(output_dir, "descriptive_statistics.csv"), index=False)

    return {
        "overview": overview_df,
        "missingness": missing_df,
        "target": target_df,
        "stage": stage_counts,
        "descriptive": desc_df
    }
