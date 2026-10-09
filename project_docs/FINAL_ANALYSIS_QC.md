# FINAL ANALYSIS QUALITY CONTROL (QC) REPORT

## 1. Input Dataset & Provenance
- **Dataset File**: `CKD_NHANES_2021_2023.csv`
- **Raw Dimensions**: 11933 rows × 29 columns.
- **Valid Analysed Cohort**: 6326 participants with complete KDIGO staging.
- **Excluded Untested/Pediatric Records**: 5607 participants with `ckd_stage == 'Unknown'`.
- **Target Variable**: `ckd_present` (Binary: 0 = No CKD [56.78%], 1 = Documented CKD [43.22%]).
- **Positive Class**: 1 (CKD presence).

---

## 2. Feature Policy & Target Leakage Elimination
- **Approved Primary Predictors (21 features)**:
  - *Demographics & SES*: `age`, `gender`, `ethnicity`, `education_level`, `poverty_income_ratio`.
  - *Vitals & Anthropometrics*: `bmi`, `weight_kg`, `height_cm`, `bp_systolic`, `bp_diastolic`.
  - *Routine Blood Labs*: `blood_urea_nitrogen`, `albumin_serum`, `phosphorus`, `bicarbonate`, `calcium`, `uric_acid`.
  - *Comorbidities & Lifestyle*: `diabetes_diagnosed`, `insulin_use`, `diabetes_pills`, `ever_smoked`, `current_smoker`.
- **Excluded Target-Defining Leakage Features (7 variables)**:
  - `egfr` (Deterministic GFR threshold).
  - `serum_creatinine` (Primary formula component for 2021 CKD-EPI equation).
  - `urine_albumin`, `urine_creatinine`, `albumin_creatinine_ratio` (Direct UACR threshold).
  - `ckd_stage` (Direct outcome staging).
  - `participant_id` (Administrative identifier).

---

## 3. Machine Learning Models & Validation Design
- **Validation Scheme**: Repeated Stratified 5-Fold Cross-Validation with 5 Repeats (25 total fold evaluations per model).
- **Leakage Safeguards**: All transformations (Median imputation, StandardScaling, OneHotEncoding) fitted strictly on each training fold.
- **Evaluated Architectures**:
  1. **Logistic Regression** (L2 regularized linear baseline).
  2. **Random Forest** (Nonlinear bagged decision trees, $n=300$).
  3. **XGBoost** (Gradient-boosted decision trees, $n=300$, depth=3, lr=0.05).

---

## 4. Key Performance Summary Across 25 Folds
| Model | ROC_AUC_Summary | PR_AUC_Summary | Sensitivity_Summary | Specificity_Summary | F1_Score_Summary | Accuracy_Summary | MCC_Summary | Brier_Score_Summary |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.844 ± 0.012 | 0.824 ± 0.014 | 0.730 ± 0.015 | 0.814 ± 0.017 | 0.740 ± 0.014 | 0.778 ± 0.012 | 0.546 ± 0.025 | 0.156 ± 0.006 |
| Random Forest | 0.841 ± 0.012 | 0.813 ± 0.016 | 0.719 ± 0.014 | 0.813 ± 0.018 | 0.732 ± 0.013 | 0.772 ± 0.012 | 0.534 ± 0.025 | 0.160 ± 0.006 |
| XGBoost | 0.846 ± 0.013 | 0.825 ± 0.015 | 0.702 ± 0.016 | 0.827 ± 0.016 | 0.728 ± 0.015 | 0.773 ± 0.013 | 0.535 ± 0.026 | 0.156 ± 0.007 |

---

## 5. SHAP Attribution Stability & Missingness Sensitivity Findings
- **Attribution Stability**: Pairwise Spearman rank correlation across 25 folds was high across all architectures, with Top-5 Jaccard overlap demonstrating substantial feature selection consistency.
- **Dominant Predictive Features**: Age, Blood Urea Nitrogen (BUN), Blood Pressure (Systolic), Uric Acid, and Diabetes Status consistently ranked as the primary clinical risk factors for CKD classification.
- **Missingness Stress Test**: Predictive performance showed gradual, graceful degradation under 10% and 20% additional synthetic MCAR missingness without catastrophic failure.

---

## 6. Execution Verification & Artifact Status
- **Unit & Integration Tests**: 100% Passed.
- **Generated Tables**: All CSV tables in `results/tables/` and `results/data_audit/` are verified non-empty.
- **Generated Figures**: All 300 DPI publication-quality figures in `results/figures/` generated successfully.
- **Artifact Manifest**: Saved in `results/metadata/artifact_manifest.csv`.

---

## 7. Scientific Limitations
1. **Cross-Sectional Provenance**: NHANES is cross-sectional with single-timepoint lab values; models classify concurrent CKD status rather than predicting future incident CKD onset.
2. **Exclusion of Direct Diagnostic Labs**: Excluding eGFR and UACR is essential to avoid trivial tautological reconstruction, but limits classification to pre-diagnostic systemic risk factors.
3. **Synthetic Missingness**: The MCAR stress test evaluates algorithmic robustness but does not model complex clinical Missing Not at Random (MNAR) patterns.
