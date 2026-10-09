"""
Visualization and publication-quality figure generation module.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
import shap

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})


def plot_target_and_stage_distribution(
    valid_df: pd.DataFrame,
    output_dir: str = "results/figures"
):
    """Plot target class and CKD stage distributions."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Binary target
    counts = valid_df["ckd_present"].value_counts()
    labels = ["No CKD (Class 0)", "CKD Present (Class 1)"]
    axes[0].bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#2b5c8f", "#d95f02"], alpha=0.85, edgecolor="black")
    axes[0].set_title("Binary Target Distribution (N=6,326)")
    axes[0].set_ylabel("Number of Participants")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        axes[0].text(i, v + 50, f"{v} ({v/len(valid_df)*100:.1f}%)", ha="center", fontweight="bold")

    # Staging breakdown
    stage_counts = valid_df["ckd_stage"].value_counts()
    sns.barplot(x=stage_counts.values, y=stage_counts.index, ax=axes[1], palette="crest", edgecolor="black")
    axes[1].set_title("KDIGO CKD Staging Breakdown in Analyzed Cohort")
    axes[1].set_xlabel("Number of Participants")
    for i, v in enumerate(stage_counts.values):
        axes[1].text(v + 30, i, f"{v} ({v/len(valid_df)*100:.1f}%)", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=300)
    plt.close()


def plot_missingness_bar(
    missing_df: pd.DataFrame,
    output_dir: str = "results/figures"
):
    """Plot feature missingness percentages in the valid cohort."""
    os.makedirs(output_dir, exist_ok=True)
    df_sorted = missing_df.sort_values(by="Valid_Missing_Pct", ascending=False)
    # Filter only features with > 0 missingness or all approved
    df_to_plot = df_sorted[df_sorted["Valid_Missing_Pct"] > 0]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_to_plot["Column"], df_to_plot["Valid_Missing_Pct"], color="#4575b4", alpha=0.85, edgecolor="black")
    plt.xlabel("Missingness Percentage (%) in Valid Cohort (N=6,326)")
    plt.title("Feature Missingness Rates")
    plt.gca().invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_missingness.png"), dpi=300)
    plt.close()


def plot_roc_and_pr_curves(
    predictions_df: pd.DataFrame,
    output_dir: str = "results/figures"
):
    """Plot cross-validation aggregated ROC and PR curves for all models."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"Logistic Regression": "#2b5c8f", "Random Forest": "#238b45", "XGBoost": "#d95f02"}

    for model_name, group in predictions_df.groupby("Model"):
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_prob"])
        prec, rec, _ = precision_recall_curve(group["y_true"], group["y_prob"])
        color = colors.get(model_name, "#333333")

        axes[0].plot(fpr, tpr, label=model_name, color=color, linewidth=2)
        axes[1].plot(rec, prec, label=model_name, color=color, linewidth=2)

    # ROC decoration
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].set_title("Cross-Validated ROC Curves")
    axes[0].legend(loc="lower right")

    # PR decoration
    base_rate = predictions_df["y_true"].mean()
    axes[1].axhline(base_rate, color="k", linestyle="--", alpha=0.6, label=f"Prevalence ({base_rate:.2f})")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (PPV)")
    axes[1].set_title("Cross-Validated Precision-Recall Curves")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_pr_curves.png"), dpi=300)
    plt.close()


def plot_model_comparison_bars(
    agg_metrics_df: pd.DataFrame,
    output_dir: str = "results/figures"
):
    """Plot comparison of models across primary performance metrics."""
    os.makedirs(output_dir, exist_ok=True)
    metrics_to_plot = ["ROC_AUC", "PR_AUC", "Sensitivity", "Specificity", "F1_Score", "Accuracy", "MCC"]

    plot_rows = []
    for _, row in agg_metrics_df.iterrows():
        model = row["Model"]
        for m in metrics_to_plot:
            if f"{m}_Mean" in row:
                plot_rows.append({
                    "Model": model,
                    "Metric": m.replace("_", " "),
                    "Mean": row[f"{m}_Mean"],
                    "SD": row[f"{m}_SD"]
                })

    plot_df = pd.DataFrame(plot_rows)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="Metric", y="Mean", hue="Model", palette=["#2b5c8f", "#238b45", "#d95f02"], edgecolor="black")
    plt.title("Model Performance Comparison Across 25 Cross-Validation Folds")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
    plt.close()


def plot_calibration_curves(
    predictions_df: pd.DataFrame,
    output_dir: str = "results/figures"
):
    """Plot probability calibration curves across evaluated models."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    colors = {"Logistic Regression": "#2b5c8f", "Random Forest": "#238b45", "XGBoost": "#d95f02"}

    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    for model_name, group in predictions_df.groupby("Model"):
        prob_true, prob_pred = calibration_curve(group["y_true"], group["y_prob"], n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=model_name, color=colors.get(model_name, "#333333"), linewidth=2)

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.title("Probability Calibration (Reliability Diagram)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_plots.png"), dpi=300)
    plt.close()


def plot_global_feature_importance(
    global_importance_by_model: Dict[str, pd.DataFrame],
    output_dir: str = "results/figures"
):
    """Plot side-by-side global feature importance rankings across models."""
    os.makedirs(output_dir, exist_ok=True)
    models = list(global_importance_by_model.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 8), sharey=False)
    if len(models) == 1:
        axes = [axes]

    colors = {"Logistic Regression": "#2b5c8f", "Random Forest": "#238b45", "XGBoost": "#d95f02"}

    for idx, model_name in enumerate(models):
        df_imp = global_importance_by_model[model_name].head(15).sort_values(by="Mean_Abs_SHAP", ascending=True)
        ax = axes[idx]
        ax.barh(df_imp["Feature"], df_imp["Mean_Abs_SHAP"], color=colors.get(model_name, "#333333"), alpha=0.85, edgecolor="black")
        ax.set_title(f"{model_name}\nTop 15 Predictors (SHAP)")
        ax.set_xlabel("Mean |SHAP Value|")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_feature_importance.png"), dpi=300)
    plt.close()


def plot_stability_heatmap(
    pairwise_stability_by_model: Dict[str, pd.DataFrame],
    n_folds: int = 25,
    output_dir: str = "results/figures"
):
    """Plot pairwise cross-fold Spearman stability heatmap for XGBoost/RF."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "XGBoost" if "XGBoost" in pairwise_stability_by_model else list(pairwise_stability_by_model.keys())[0]
    df_pair = pairwise_stability_by_model[model_name]

    matrix = np.ones((n_folds, n_folds))
    for _, row in df_pair.iterrows():
        i, j = int(row["Fold_A"]), int(row["Fold_B"])
        val = row["Spearman_Rank_Correlation"]
        matrix[i, j] = val
        matrix[j, i] = val

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="viridis", vmin=0.7, vmax=1.0, cbar_kws={"label": "Spearman Rank Correlation"})
    plt.title(f"SHAP Attribution Stability Heatmap Across {n_folds} Folds ({model_name})")
    plt.xlabel("Cross-Validation Fold Index")
    plt.ylabel("Cross-Validation Fold Index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_stability_heatmap.png"), dpi=300)
    plt.close()


def plot_feature_rank_variability(
    variability_by_model: Dict[str, pd.DataFrame],
    output_dir: str = "results/figures"
):
    """Plot rank distribution and standard deviation across folds."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 7))

    for model_name, df_var in variability_by_model.items():
        top_feats = df_var.head(12)
        plt.errorbar(
            top_feats["Feature"],
            top_feats["Mean_Rank"],
            yerr=top_feats["SD_Rank"],
            fmt="o-",
            capsize=4,
            label=model_name
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Feature Rank (Lower = More Important)")
    plt.title("Feature Rank Variability Across 25 Resampled Folds (Mean ± SD)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_rank_variability.png"), dpi=300)
    plt.close()


def plot_missingness_sensitivity_curves(
    missingness_df: pd.DataFrame,
    output_dir: str = "results/figures"
):
    """Plot predictive degradation and explanation stability under missingness stress tests."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    agg_df = missingness_df.groupby(["Model", "Masking_Level"]).agg(
        ROC_AUC_Mean=("ROC_AUC", "mean"),
        ROC_AUC_SD=("ROC_AUC", "std"),
        SHAP_Stability_Mean=("SHAP_Rank_Correlation_With_Baseline", "mean"),
        SHAP_Stability_SD=("SHAP_Rank_Correlation_With_Baseline", "std")
    ).reset_index()

    colors = {"Logistic Regression": "#2b5c8f", "Random Forest": "#238b45", "XGBoost": "#d95f02"}

    # Performance degradation
    for model_name, group in agg_df.groupby("Model"):
        axes[0].errorbar(
            group["Masking_Level"] * 100,
            group["ROC_AUC_Mean"],
            yerr=group["ROC_AUC_SD"],
            marker="o",
            capsize=4,
            linewidth=2,
            label=model_name,
            color=colors.get(model_name, "#333333")
        )

    axes[0].set_xlabel("Additional Imposed Missingness (%)")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Predictive Performance Under Missingness Stress Test")
    axes[0].legend(loc="lower left")

    # Explanation stability degradation
    for model_name, group in agg_df.groupby("Model"):
        axes[1].errorbar(
            group["Masking_Level"] * 100,
            group["SHAP_Stability_Mean"],
            yerr=group["SHAP_Stability_SD"],
            marker="s",
            capsize=4,
            linewidth=2,
            label=model_name,
            color=colors.get(model_name, "#333333")
        )

    axes[1].set_xlabel("Additional Imposed Missingness (%)")
    axes[1].set_ylabel("Spearman Rank Correlation vs Baseline SHAP")
    axes[1].set_title("Explanation Stability Under Missingness Stress Test")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missingness_sensitivity.png"), dpi=300)
    plt.close()
