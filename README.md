# Explainable Machine Learning for Chronic Kidney Disease Classification Using NHANES 2021–2023

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

An end-to-end reproducible research pipeline evaluating **explainable machine learning models (Logistic Regression, Random Forest, XGBoost)** for Chronic Kidney Disease (CKD) risk-factor classification using the **NHANES 2021–2023** cohort.

This investigation rigorously adheres to **PROBAST+AI (2025)** and **TRIPOD+AI (2024)** reporting guidelines, prevents tautological target leakage (excluding direct KDIGO criterion variables), quantifies **SHAP feature attribution stability across 25 cross-validation folds (300 pairwise comparisons)**, and performs **controlled MCAR missing-data stress testing**.

---

## 📌 Clinical & Methodological Scope

- **Study Type**: Observational cross-sectional risk-factor classification study.
- **Data Source**: Continuous National Health and Nutrition Examination Survey (NHANES) 2021–2023 cycle ($N=11,933$ raw participants).
- **Target Population**: Adults and adolescents ($\ge 12$ years) with verified KDIGO kidney status ($N=6,326$ valid participants: 3,592 non-CKD [56.8%], 2,734 CKD [43.2%]). Pediatric/untested participants with missing eGFR/creatinine (`ckd_stage == 'Unknown'`, $N=5,607$) are excluded to prevent target misclassification.
- **Leakage Prevention**: All direct target-defining components (`egfr`, `serum_creatinine`, `urine_albumin`, `urine_creatinine`, `albumin_creatinine_ratio`, `ckd_stage`, `participant_id`) are strictly excluded from the predictor matrix.
- **Predictor Set**: 21 non-leaking demographic, anthropometric, socioeconomic, behavioral, and clinical biomarkers.
- **Disclaimer**: This is a risk-factor classification model, **not** an automated diagnostic tool, prospective onset predictor, or causal inference engine.

---

## 📁 Repository Structure

```
.
├── config/
│   └── project.yaml              # Central YAML configuration (paths, CV, hyperparameters)
├── project_docs/
│   ├── PROJECT_CONTEXT.md        # Synthesis of study design, RQ1-4, and KDIGO/TRIPOD standards
│   ├── DATASET_AUDIT.md          # Full feature schema, missingness, and target audit
│   ├── IMPLEMENTATION_STATUS.md  # Detailed implementation and execution status log
│   └── FINAL_ANALYSIS_QC.md      # Comprehensive scientific quality control and results synthesis
├── references/                   # Reference literature PDFs (KDIGO 2024, PROBAST+AI, etc.)
├── results/
│   ├── data_audit/               # Baseline feature audit tables (CSV/JSON)
│   ├── explanations/             # Aggregated SHAP matrices, fold attributions, importance rankings
│   ├── figures/                  # Publication-quality figures (300 DPI PNGs)
│   ├── metadata/                 # Environment manifest, config snapshots, artifact manifests
│   ├── metrics/                  # Fold-level and aggregated CV performance metrics
│   ├── missingness_sensitivity/  # 10% and 20% MCAR stress test metrics and rank shifts
│   ├── predictions/              # Out-of-fold calibrated probabilities and test indices
│   ├── stability/                # 300 fold-pair Spearman correlations and Top-k Jaccard indices
│   └── tables/                   # Publication-ready summary tables (CSV)
├── src/
│   ├── __init__.py
│   ├── data_loading.py           # Cohort extraction and schema validation
│   ├── data_audit.py             # Feature statistics, distributions, and missingness audit
│   ├── leakage_audit.py          # KDIGO target leakage verification
│   ├── preprocessing.py          # Leakage-safe ColumnTransformer pipelines
│   ├── models.py                 # LR, RF, XGBoost initializers
│   ├── metrics.py                # ROC-AUC, PR-AUC, Calibration, Brier score, net benefit
│   ├── explainability.py         # Native TreeSHAP / LinearSHAP and parent-feature aggregation
│   ├── stability_analysis.py     # Pairwise Spearman correlation & Jaccard index computation
│   ├── missingness_sensitivity.py# MCAR masking stress-testing engine
│   ├── visualizations.py         # Publication-grade plotting suite (ROC, PR, SHAP, Heatmaps)
│   ├── reporting.py              # Publication table formatting and artifact manifest builder
│   └── run_pipeline.py           # Central orchestration entry point
├── tests/
│   ├── test_data_loading.py      # Unit tests for loading and cohort filtering
│   ├── test_preprocessing.py     # Unit tests for training-fold-only fitting
│   ├── test_models.py            # Unit tests for model predictions and metrics
│   ├── test_explainability.py    # Unit tests for SHAP value computation and shape preservation
│   └── test_stability.py         # Unit tests for Spearman and Jaccard stability metrics
├── requirements.txt              # Primary project dependencies
└── README.md                     # Project overview and reproduction instructions
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/username/CKD.git
cd CKD

# Install dependencies (or use local deps directory)
pip install -r requirements.txt
```

### 2. Run Test Suite

Verify all modular components and leakage protections before execution:

```bash
PYTHONPATH=deps:. pytest tests/ -v
```

### 3. Execute Complete Analysis Pipeline

Run all 7 stages (Data extraction, Audit, 25-Fold CV, SHAP attribution, Stability analysis, Missingness stress tests, Visualizations & Reporting):

```bash
PYTHONPATH=deps:. python3 src/run_pipeline.py
```

---

## 📊 Key Research Findings

### 1. Model Discrimination (Repeated Stratified 5-Fold CV, 25 Folds)
| Model | ROC-AUC (Mean ± SD) | PR-AUC (Mean ± SD) | Accuracy (Mean ± SD) | Brier Score (Mean ± SD) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.846 ± 0.013** | **0.825 ± 0.015** | 0.773 ± 0.013 | 0.158 ± 0.007 |
| **Logistic Regression** | 0.844 ± 0.012 | 0.824 ± 0.014 | **0.778 ± 0.012** | **0.156 ± 0.006** |
| **Random Forest** | 0.841 ± 0.012 | 0.813 ± 0.016 | 0.772 ± 0.012 | 0.160 ± 0.006 |

### 2. Top-5 Consensus Risk Factors (SHAP Global Attribution)
1. **Age** (Mean |SHAP| ~0.14–0.17; dominant monotonic positive risk factor)
2. **Systolic Blood Pressure / High Blood Pressure History**
3. **Glycohemoglobin (HbA1c)**
4. **Blood Urea Nitrogen (BUN)**
5. **Uric Acid**

### 3. Explanation Stability Across 300 Fold Pairs
- **Random Forest**: Spearman $\rho = 0.965 \pm 0.015$; Top-5 Jaccard $= 1.000 \pm 0.000$; Top-10 Jaccard $= 0.958 \pm 0.078$.
- **XGBoost**: Spearman $\rho = 0.941 \pm 0.021$; Top-5 Jaccard $= 1.000 \pm 0.000$; Top-10 Jaccard $= 0.800 \pm 0.109$.
- **Logistic Regression**: Spearman $\rho = 0.897 \pm 0.055$; Top-5 Jaccard $= 0.927 \pm 0.138$; Top-10 Jaccard $= 0.816 \pm 0.102$.

### 4. Missing-Data Stress Testing
- Under **10% additional MCAR missingness**, ROC-AUC degradation is $<0.015$ across all models, and Top-5 feature attribution ranking remains intact (Spearman rank correlation $>0.93$).
- Under **20% additional MCAR missingness**, ROC-AUC drops by $\sim 0.025–0.035$, demonstrating robust graceful degradation without catastrophic failure.

---

## 📜 Citations & Reference Standards

1. **KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease**: *Kidney International*, 2024.
2. **Inker LA, et al.**: *New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race*. N Engl J Med, 2021.
3. **PROBAST+AI (2025)**: Risk of Bias Assessment Tool for Prediction Models using Artificial Intelligence.
4. **TRIPOD+AI (2024)**: Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis.
5. **Lundberg SM, Lee SI**: *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017.
