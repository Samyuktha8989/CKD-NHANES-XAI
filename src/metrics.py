"""
Performance metric calculations and evaluation module for binary CKD classification.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix
)

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)


def calculate_binary_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive binary classification metrics.
    Handles mathematical edge cases explicitly.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Discrimination metrics
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = np.nan

    # Classification metrics
    sensitivity = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0

    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1_Score": f1,
        "Accuracy": accuracy,
        "MCC": mcc,
        "Brier_Score": brier,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }


def aggregate_cv_metrics(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Mean ± SD across evaluated cross-validation folds.
    """
    metric_cols = [
        "ROC_AUC", "PR_AUC", "Sensitivity", "Specificity",
        "Precision", "F1_Score", "Accuracy", "MCC", "Brier_Score"
    ]

    summary_rows = []
    for model_name, group in fold_metrics_df.groupby("Model"):
        row = {"Model": model_name, "Evaluated_Folds": len(group)}
        for metric in metric_cols:
            if metric in group.columns:
                mean_val = group[metric].mean()
                std_val = group[metric].std()
                row[f"{metric}_Mean"] = round(mean_val, 4)
                row[f"{metric}_SD"] = round(std_val, 4)
                row[f"{metric}_Summary"] = f"{mean_val:.3f} ± {std_val:.3f}"
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
