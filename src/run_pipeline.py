"""
Master Pipeline Orchestrator for Explainable ML for CKD Classification.
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

# Ensure deps is in sys.path
deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)

from src.data_loading import load_config, load_raw_dataset, filter_valid_cohort, prepare_feature_target_split
from src.data_audit import run_comprehensive_audit
from src.leakage_audit import run_leakage_audit
from src.models import get_all_models
from src.metrics import aggregate_cv_metrics
from src.validation import run_repeated_cv
from src.stability_analysis import (
    compute_pairwise_stability_metrics,
    summarize_stability_distributions,
    compute_feature_rank_variability,
    compute_cross_model_agreement
)
from src.missingness_sensitivity import evaluate_missingness_stress_test
from src.visualizations import (
    plot_target_and_stage_distribution,
    plot_missingness_bar,
    plot_roc_and_pr_curves,
    plot_model_comparison_bars,
    plot_calibration_curves,
    plot_global_feature_importance,
    plot_stability_heatmap,
    plot_feature_rank_variability,
    plot_missingness_sensitivity_curves
)
from src.reporting import (
    save_environment_versions,
    generate_artifact_manifest,
    generate_final_qc_report
)


def run_full_pipeline(config_path: str = "config/project.yaml"):
    """Execute the end-to-end CKD analysis pipeline."""
    start_time = time.time()
    print("=" * 80)
    print("STARTING EXPLAINABLE ML FOR CKD CLASSIFICATION PIPELINE")
    print("=" * 80)

    # 1. Load configuration
    config = load_config(config_path)
    output_dirs = {
        "results_dir": config["paths"]["results_dir"],
        "data_audit_dir": config["paths"]["data_audit_dir"],
        "tables_dir": config["paths"]["tables_dir"],
        "figures_dir": config["paths"]["figures_dir"],
        "metrics_dir": config["paths"]["metrics_dir"],
        "predictions_dir": config["paths"]["predictions_dir"],
        "explanations_dir": config["paths"]["explanations_dir"],
        "stability_dir": config["paths"]["stability_dir"],
        "missingness_dir": config["paths"]["missingness_dir"],
        "models_dir": config["paths"]["models_dir"],
        "metadata_dir": config["paths"]["metadata_dir"]
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)

    # 2. Save environment versions
    save_environment_versions(output_dirs["metadata_dir"])

    # 3. Data Loading & Cohort Definition
    dataset_path = config["data"]["raw_dataset_path"]
    print(f"\n[Stage 1/7] Loading dataset from {dataset_path}...")
    raw_df = load_raw_dataset(dataset_path)
    print(f"  Raw dataset loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    valid_df, cohort_info = filter_valid_cohort(
        df=raw_df,
        stage_col=config["data"]["stage_column"],
        invalid_stages=config["data"]["filter_invalid_stages"]
    )
    print(f"  Valid cohort extracted: {len(valid_df)} records (Excluded {cohort_info['filtered_unknown_records']} unknown-stage rows)")

    # 4. Data Audit & Leakage Assessment
    print("\n[Stage 2/7] Running comprehensive data audit and leakage checks...")
    audit_results = run_comprehensive_audit(raw_df, valid_df, output_dirs["data_audit_dir"])
    leakage_df = run_leakage_audit(valid_df, config, output_dirs["data_audit_dir"])
    print("  Data audit and leakage tables saved.")

    # 5. Extract Feature Matrix & Target
    X, y, num_feats, cat_feats = prepare_feature_target_split(valid_df, config)
    print(f"  Approved predictor matrix: {X.shape[1]} features ({len(num_feats)} numeric, {len(cat_feats)} categorical)")

    # 6. Initialize Models
    models = get_all_models(config, seed=config["validation"]["random_state"])
    print(f"  Models initialized: {list(models.keys())}")

    # 7. Run Repeated Stratified 5-Fold Cross-Validation (25 Folds)
    print("\n[Stage 3/7] Executing 25-Fold Cross-Validation and SHAP explanations...")
    cv_results = run_repeated_cv(
        X=X,
        y=y,
        numeric_features=num_feats,
        categorical_features=cat_feats,
        models=models,
        config=config,
        output_dirs=output_dirs
    )

    fold_metrics_df = cv_results["fold_metrics"]
    predictions_df = cv_results["predictions"]
    fold_importance_by_model = cv_results["fold_importance_by_model"]

    # Save aggregated model performance
    agg_metrics_df = aggregate_cv_metrics(fold_metrics_df)
    agg_metrics_df.to_csv(os.path.join(output_dirs["tables_dir"], "model_comparison_table.csv"), index=False)
    print("\n  Aggregated Cross-Validation Performance:")
    print(agg_metrics_df[["Model", "ROC_AUC_Summary", "PR_AUC_Summary", "F1_Score_Summary", "Accuracy_Summary"]].to_string(index=False))

    # Confusion matrix summary table
    cm_summary = fold_metrics_df.groupby("Model")[["TP", "FP", "TN", "FN"]].mean().reset_index()
    cm_summary.to_csv(os.path.join(output_dirs["tables_dir"], "confusion_matrix_summary.csv"), index=False)

    # 8. SHAP Attribution Stability Analysis
    print("\n[Stage 4/7] Computing SHAP attribution stability and feature rank variability...")
    global_importance_by_model = {}
    pairwise_stability_by_model = {}
    stability_summaries = []
    variability_by_model = {}

    for model_name, fold_dfs in fold_importance_by_model.items():
        # Global aggregated feature importance across all 25 folds
        combined_folds_df = pd.concat(fold_dfs, ignore_index=True)
        global_imp = combined_folds_df.groupby("Feature")["Mean_Abs_SHAP"].mean().reset_index()
        global_imp = global_imp.sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        global_imp["Rank"] = np.arange(1, len(global_imp) + 1)
        global_imp["Model"] = model_name
        global_importance_by_model[model_name] = global_imp
        global_imp.to_csv(os.path.join(output_dirs["explanations_dir"], f"global_shap_{model_name.lower().replace(' ', '_')}.csv"), index=False)

        # Pairwise stability across 300 fold pairs
        pairwise_df = compute_pairwise_stability_metrics(fold_dfs, top_k_list=[5, 10])
        pairwise_df["Model"] = model_name
        pairwise_stability_by_model[model_name] = pairwise_df
        pairwise_df.to_csv(os.path.join(output_dirs["stability_dir"], f"pairwise_stability_{model_name.lower().replace(' ', '_')}.csv"), index=False)

        # Stability distribution summary
        stab_summary = summarize_stability_distributions(pairwise_df, model_name, top_k_list=[5, 10])
        stability_summaries.append(stab_summary)

        # Feature rank variability
        var_df = compute_feature_rank_variability(fold_dfs)
        var_df["Model"] = model_name
        variability_by_model[model_name] = var_df
        var_df.to_csv(os.path.join(output_dirs["stability_dir"], f"feature_variability_{model_name.lower().replace(' ', '_')}.csv"), index=False)

    stability_summary_df = pd.DataFrame(stability_summaries)
    stability_summary_df.to_csv(os.path.join(output_dirs["tables_dir"], "stability_summary_table.csv"), index=False)
    print("\n  Explanation Stability Summary (Spearman Rank Correlation & Top-k Jaccard):")
    print(stability_summary_df.to_string(index=False))

    # Cross-model ranking agreement
    cross_model_df = compute_cross_model_agreement(global_importance_by_model)
    cross_model_df.to_csv(os.path.join(output_dirs["tables_dir"], "cross_model_agreement.csv"), index=False)

    # 9. Missingness Sensitivity Stress Testing
    print("\n[Stage 5/7] Running controlled missingness stress tests (10% and 20% masking)...")
    first_fold_pipelines = cv_results["first_fold_pipelines"]
    missingness_results_all = []

    # Held-out validation split of fold 0 for stress testing
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx0, val_idx0 = next(skf.split(X, y))
    X_val0, y_val0 = X.iloc[val_idx0], y.iloc[val_idx0]

    for model_name, (fold_model, preprocessor, tf_names, col_to_orig_map) in first_fold_pipelines.items():
        base_imp = global_importance_by_model[model_name]
        stress_res = evaluate_missingness_stress_test(
            model=fold_model,
            model_name=model_name,
            fitted_preprocessor=preprocessor,
            X_test_raw=X_val0,
            y_test=y_val0,
            baseline_importance_df=base_imp,
            transformed_feature_names=tf_names,
            column_to_original_map=col_to_orig_map,
            original_feature_order=num_feats + cat_feats,
            masking_levels=config["missingness_sensitivity"]["masking_levels"],
            masking_seeds=config["missingness_sensitivity"]["masking_seeds"],
            shap_sample_limit=config["explainability"]["shap_sample_limit"]
        )
        missingness_results_all.extend(stress_res)

    missingness_df = pd.DataFrame(missingness_results_all)
    missingness_df.to_csv(os.path.join(output_dirs["missingness_dir"], "missingness_stress_test_results.csv"), index=False)

    # Aggregate missingness summary
    agg_missingness = missingness_df.groupby(["Model", "Masking_Level"]).agg(
        ROC_AUC_Mean=("ROC_AUC", "mean"),
        ROC_AUC_SD=("ROC_AUC", "std"),
        F1_Score_Mean=("F1_Score", "mean"),
        F1_Score_SD=("F1_Score", "std"),
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_SD=("Accuracy", "std"),
        SHAP_Stability_Mean=("SHAP_Rank_Correlation_With_Baseline", "mean"),
        SHAP_Stability_SD=("SHAP_Rank_Correlation_With_Baseline", "std")
    ).reset_index()
    agg_missingness.to_csv(os.path.join(output_dirs["tables_dir"], "missingness_summary_table.csv"), index=False)

    # 10. Publication Figure Generation
    print("\n[Stage 6/7] Generating publication-quality figures (300 DPI)...")
    plot_target_and_stage_distribution(valid_df, output_dirs["figures_dir"])
    plot_missingness_bar(audit_results["missingness"], output_dirs["figures_dir"])
    plot_roc_and_pr_curves(predictions_df, output_dirs["figures_dir"])
    plot_model_comparison_bars(agg_metrics_df, output_dirs["figures_dir"])
    plot_calibration_curves(predictions_df, output_dirs["figures_dir"])
    plot_global_feature_importance(global_importance_by_model, output_dirs["figures_dir"])
    plot_stability_heatmap(pairwise_stability_by_model, n_folds=25, output_dir=output_dirs["figures_dir"])
    plot_feature_rank_variability(variability_by_model, output_dirs["figures_dir"])
    plot_missingness_sensitivity_curves(missingness_df, output_dirs["figures_dir"])
    print("  All figures generated in results/figures/.")

    # 11. Reporting, Artifact Manifest & QC Report
    print("\n[Stage 7/7] Generating artifact manifest and Final Quality Control Report...")
    # Format metrics markdown table
    cols_to_show = ["Model", "ROC_AUC_Summary", "PR_AUC_Summary", "Sensitivity_Summary", "Specificity_Summary", "F1_Score_Summary", "Accuracy_Summary", "MCC_Summary", "Brier_Score_Summary"]
    sub_df = agg_metrics_df[[c for c in cols_to_show if c in agg_metrics_df.columns]]
    headers = list(sub_df.columns)
    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join([":---:" if i > 0 else ":---" for i in range(len(headers))]) + " |"]
    for _, r in sub_df.iterrows():
        md_lines.append("| " + " | ".join(str(r[c]) for c in headers) + " |")
    metrics_markdown = "\n".join(md_lines)

    summary_dict = {
        "dataset_file": dataset_path,
        "raw_rows": raw_df.shape[0],
        "raw_cols": raw_df.shape[1],
        "valid_rows": valid_df.shape[0],
        "excluded_rows": cohort_info["filtered_unknown_records"],
        "total_folds": len(fold_metrics_df) // len(models),
        "metrics_table_markdown": metrics_markdown,
        "execution_time_seconds": round(time.time() - start_time, 2)
    }

    with open(os.path.join(output_dirs["metadata_dir"], "run_summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)

    manifest_df = generate_artifact_manifest(output_dirs["results_dir"], os.path.join(output_dirs["metadata_dir"], "artifact_manifest.csv"))
    generate_final_qc_report(summary_dict, "project_docs/FINAL_ANALYSIS_QC.md")

    # Update implementation status document
    with open("project_docs/IMPLEMENTATION_STATUS.md", "a") as f:
        f.write("\n\n## Pipeline Execution Completed\n- Status: Success\n- Total Duration: " + str(summary_dict["execution_time_seconds"]) + " seconds\n- Generated Artifacts: " + str(len(manifest_df)) + " files\n")

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {summary_dict['execution_time_seconds']} SECONDS")
    print(f"Total Artifacts Generated: {len(manifest_df)}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
