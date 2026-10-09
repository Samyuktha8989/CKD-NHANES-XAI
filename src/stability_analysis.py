"""
SHAP attribution stability and cross-fold explanation consistency analysis module.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy.stats import spearmanr

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)


def compute_pairwise_stability_metrics(
    fold_importance_dfs: List[pd.DataFrame],
    top_k_list: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Compute pairwise Spearman rank correlation and top-k Jaccard similarities
    across all pairs of evaluated cross-validation folds.
    """
    n_folds = len(fold_importance_dfs)
    pairwise_records = []

    for i in range(n_folds):
        df_i = fold_importance_dfs[i].set_index("Feature")
        for j in range(i + 1, n_folds):
            df_j = fold_importance_dfs[j].set_index("Feature")

            # Align on common features
            common_feats = df_i.index.intersection(df_j.index)
            ranks_i = df_i.loc[common_feats, "Rank"]
            ranks_j = df_j.loc[common_feats, "Rank"]

            spearman_corr, _ = spearmanr(ranks_i, ranks_j)

            record = {
                "Fold_A": i,
                "Fold_B": j,
                "Spearman_Rank_Correlation": float(spearman_corr)
            }

            for k in top_k_list:
                top_i = set(df_i.sort_values(by="Rank").index[:k])
                top_j = set(df_j.sort_values(by="Rank").index[:k])
                intersection = len(top_i.intersection(top_j))
                union = len(top_i.union(top_j))
                jaccard = intersection / union if union > 0 else 1.0
                record[f"Top_{k}_Jaccard"] = float(jaccard)

            pairwise_records.append(record)

    return pd.DataFrame(pairwise_records)


def summarize_stability_distributions(
    pairwise_df: pd.DataFrame,
    model_name: str,
    top_k_list: List[int] = [5, 10]
) -> Dict[str, Any]:
    """
    Compute descriptive statistics (Mean, SD, Median, IQR) for pairwise stability metrics.
    """
    summary = {
        "Model": model_name,
        "Comparisons_Count": len(pairwise_df),
        "Spearman_Mean": round(pairwise_df["Spearman_Rank_Correlation"].mean(), 4),
        "Spearman_SD": round(pairwise_df["Spearman_Rank_Correlation"].std(), 4),
        "Spearman_Median": round(pairwise_df["Spearman_Rank_Correlation"].median(), 4),
        "Spearman_IQR": round(pairwise_df["Spearman_Rank_Correlation"].quantile(0.75) - pairwise_df["Spearman_Rank_Correlation"].quantile(0.25), 4)
    }

    for k in top_k_list:
        col = f"Top_{k}_Jaccard"
        summary[f"{col}_Mean"] = round(pairwise_df[col].mean(), 4)
        summary[f"{col}_SD"] = round(pairwise_df[col].std(), 4)
        summary[f"{col}_Median"] = round(pairwise_df[col].median(), 4)
        summary[f"{col}_IQR"] = round(pairwise_df[col].quantile(0.75) - pairwise_df[col].quantile(0.25), 4)

    return summary


def compute_feature_rank_variability(
    fold_importance_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate rank variability (mean rank, std rank, min rank, max rank)
    and mean attribution variability across folds for each feature.
    """
    combined_list = []
    for fold_idx, df in enumerate(fold_importance_dfs):
        df_copy = df.copy()
        df_copy["Fold"] = fold_idx
        combined_list.append(df_copy)

    combined_df = pd.concat(combined_list, ignore_index=True)

    variability_df = combined_df.groupby("Feature").agg(
        Mean_Rank=("Rank", "mean"),
        SD_Rank=("Rank", "std"),
        Min_Rank=("Rank", "min"),
        Max_Rank=("Rank", "max"),
        Mean_SHAP=("Mean_Abs_SHAP", "mean"),
        SD_SHAP=("Mean_Abs_SHAP", "std")
    ).reset_index()

    variability_df["CV_SHAP"] = (variability_df["SD_SHAP"] / variability_df["Mean_SHAP"]).round(4)
    variability_df = variability_df.sort_values(by="Mean_Rank").reset_index(drop=True)
    return variability_df


def compute_cross_model_agreement(
    global_importance_by_model: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate cross-model Spearman rank correlations on global feature rankings.
    """
    model_names = list(global_importance_by_model.keys())
    cross_records = []

    for i in range(len(model_names)):
        m1 = model_names[i]
        df1 = global_importance_by_model[m1].set_index("Feature")
        for j in range(i + 1, len(model_names)):
            m2 = model_names[j]
            df2 = global_importance_by_model[m2].set_index("Feature")

            common = df1.index.intersection(df2.index)
            corr, _ = spearmanr(df1.loc[common, "Rank"], df2.loc[common, "Rank"])
            top5_1 = set(df1.sort_values(by="Rank").index[:5])
            top5_2 = set(df2.sort_values(by="Rank").index[:5])
            jaccard5 = len(top5_1.intersection(top5_2)) / len(top5_1.union(top5_2))

            cross_records.append({
                "Model_1": m1,
                "Model_2": m2,
                "Spearman_Rank_Correlation": round(float(corr), 4),
                "Top_5_Jaccard": round(float(jaccard5), 4)
            })

    return pd.DataFrame(cross_records)
