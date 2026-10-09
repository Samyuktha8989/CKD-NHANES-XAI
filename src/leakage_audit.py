"""
Target leakage analysis and audit module.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)


def run_leakage_audit(
    valid_df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: str = "results/data_audit"
) -> pd.DataFrame:
    """
    Audit potential target leakage among all variables in the valid cohort.
    Calculates point-biserial correlation with ckd_present for numerical variables
    and documents mathematical derivation risk.
    """
    os.makedirs(output_dir, exist_ok=True)
    target = valid_df["ckd_present"]

    excluded_leakage = config["features"].get("excluded_leakage", [])
    excluded_ids = config["features"].get("excluded_identifiers", [])
    approved_predictors = config["features"]["numeric_features"] + config["features"]["categorical_features"]

    audit_rows = []
    for col in valid_df.columns:
        if col in ["ckd_present"]:
            continue

        dtype = str(valid_df[col].dtype)
        is_num = pd.api.types.is_numeric_dtype(valid_df[col])
        corr_val = np.nan
        if is_num:
            # point-biserial / Pearson correlation with binary target
            non_null = valid_df[[col, "ckd_present"]].dropna()
            if len(non_null) > 1 and non_null[col].std() > 0:
                corr_val = non_null[col].corr(non_null["ckd_present"])

        if col in excluded_ids:
            risk_level = "None (ID)"
            decision = "Excluded"
            reason = "Administrative identifier, not a medical or biological feature."
        elif col in excluded_leakage:
            risk_level = "Severe / Direct Target Leakage"
            decision = "Excluded"
            reason = "Direct mathematical component or derivative of KDIGO CKD diagnostic definition."
        elif col in approved_predictors:
            risk_level = "Low (Approved Clinical Predictor)"
            decision = "Included"
            reason = "Pre-diagnostic risk factor, comorbidity, anthropometric, or routine non-diagnostic laboratory test."
        else:
            risk_level = "Unspecified"
            decision = "Excluded"
            reason = "Not listed in approved configuration."

        audit_rows.append({
            "Feature": col,
            "DataType": dtype,
            "TargetCorrelation": round(corr_val, 4) if pd.notnull(corr_val) else "N/A",
            "LeakageRiskLevel": risk_level,
            "Decision": decision,
            "Rationale": reason
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(os.path.join(output_dir, "leakage_audit.csv"), index=False)
    return audit_df
