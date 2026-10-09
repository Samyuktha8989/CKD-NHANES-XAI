# PROJECT CONTEXT: Explainable Machine Learning for CKD Classification Using NHANES 2021–2023

## 1. Study Overview & Working Title
- **Working Title**: "Explainable Machine Learning for Chronic Kidney Disease Classification Using NHANES 2021–2023 Data: Feature Attribution Stability and Missing-Data Sensitivity"
- **Primary Research Objective**: Evaluate the classification performance, explainability (SHAP feature attribution), attribution stability across repeated cross-validation folds, and sensitivity to synthetic missingness across three canonical machine-learning architectures (Logistic Regression, Random Forest, and XGBoost) on a nationally representative, cross-sectional population sample (NHANES 2021–2023).
- **Study Type**: Retrospective, cross-sectional epidemiological/tabular machine-learning classification and methodological robustness study.
- **Scientific Caveat**: Because NHANES is a cross-sectional survey with single-timepoint laboratory measurements, this study investigates *classification/discrimination* of concurrent CKD status and attribution stability. It is **not** a study of future disease onset, longitudinal progression, causal mechanisms, or clinical diagnosis.

---

## 2. Research Questions (RQs)
| ID | Research Question | Analytical Method |
|---|---|---|
| **RQ1** | How does discrimination, classification, and probability calibration vary across Logistic Regression, Random Forest, and XGBoost on clinical and demographic risk factors? | Repeated Stratified 5-Fold Cross-Validation (5 repeats = 25 folds); ROC-AUC, PR-AUC, Sensitivity, Specificity, Precision, F1, Accuracy, MCC, Brier Score. |
| **RQ2** | How consistent are SHAP-derived global feature rankings across resampled cross-validation folds? | Pairwise Spearman rank correlation across 25 folds; Top-5 and Top-10 Jaccard similarity indices; feature rank variance. |
| **RQ3** | Do linear, bagged-tree, and boosted-tree models identify concordant sets of influential original clinical predictors? | Cross-model Spearman rank correlations and top-$k$ feature overlap after aggregating one-hot encoded variables back to original clinical features. |
| **RQ4** | How do predictive performance and explanation rankings degrade under controlled missing-data stress tests (10% and 20% additional feature masking)? | Synthetic MCAR masking on held-out test inputs evaluated using already-fitted training-fold imputers and classifiers across multiple random seeds. |

---

## 3. Dataset Provenance & Target Construction
- **Dataset**: Kaggle "CKD-NHANES 2021–2023: Staged Kidney Disease" (`CKD_NHANES_2021_2023.csv`), derived from the CDC National Center for Health Statistics (NCHS) NHANES 2021–2023 Continuous Survey Cycle.
- **Raw Dimensions**: 11,933 participants, 29 variables.
- **Target Construction Audit**:
  - The dataset defines `ckd_stage` following KDIGO guidelines using 2021 CKD-EPI eGFR and Urinary Albumin-to-Creatinine Ratio (UACR).
  - `ckd_stage == 'No CKD'` corresponds to eGFR $\ge 90\text{ mL/min/1.73 m}^2$ and UACR $< 30\text{ mg/g}$ ($N=3,592$, coded as `ckd_present = 0`).
  - Stages 1–5 represent documented CKD ($N=2,734$, coded as `ckd_present = 1`).
  - $N=5,607$ participants have `ckd_stage == 'Unknown'` because serum creatinine/eGFR was not measured (e.g., pediatric participants $<12$ years or non-laboratory sample). The raw CSV labeled these as `ckd_present = 1` due to a boolean inequality `(ckd_stage != 'No CKD')`.
  - **Inclusion Criterion**: The evaluable cohort consists of all participants with verified, non-missing CKD status ($N=6,326$: 3,592 non-CKD [56.8%], 2,734 CKD [43.2%]).

---

## 4. Target Leakage Prevention & Feature Selection
- **Leakage Prevention**: To prevent deterministic target reconstruction (tautological prediction), all direct diagnostic markers and mathematical derivatives of the KDIGO definition must be excluded from the risk-factor predictor set:
  - Excluded target-defining features: `egfr`, `serum_creatinine`, `urine_albumin`, `urine_creatinine`, `albumin_creatinine_ratio`, `ckd_stage`.
  - Excluded administrative features: `participant_id`.
- **Approved Primary Predictor Set (21 features)**:
  - *Demographics*: `age`, `gender`, `ethnicity`, `education_level`, `poverty_income_ratio`.
  - *Anthropometrics & Vitals*: `bmi`, `weight_kg`, `height_cm`, `bp_systolic`, `bp_diastolic`.
  - *Biochemistry / Routine Labs*: `blood_urea_nitrogen`, `albumin_serum`, `phosphorus`, `bicarbonate`, `calcium`, `uric_acid`.
  - *Comorbidities & Lifestyle*: `diabetes_diagnosed`, `insulin_use`, `diabetes_pills`, `ever_smoked`, `current_smoker`.

---

## 5. Methodological Framework & Safeguards
1. **Validation Design**: Repeated Stratified 5-Fold Cross-Validation with 5 repeats ($5 \times 5 = 25$ independent fold evaluations per model).
2. **Leakage-Safe Preprocessing**:
   - Numeric variables: Median imputation + StandardScaler (fitted strictly on training fold).
   - Categorical variables: Most-frequent imputation + OneHotEncoder (handle_unknown='ignore', fitted strictly on training fold).
   - Validation fold data is purely transformed using the training fold's fitted pipeline.
3. **Model Configurations**:
   - Logistic Regression: L2 penalty, $C=1.0$, `max_iter=2000`, `random_state=seed`.
   - Random Forest: `n_estimators=300`, `min_samples_leaf=2`, `max_features='sqrt'`, `random_state=seed`.
   - XGBoost: `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='logloss'`, `random_state=seed`.
4. **Explainability & Stability**:
   - Explanations calculated via SHAP on held-out validation folds.
   - One-hot encoded feature attributions aggregated back to original parent clinical features using absolute sum attribution before rank stability computations.
5. **Missingness Stress Testing**:
   - 10% and 20% MCAR masking on observed test-fold features across 5 random seeds.
   - Evaluated directly through training-fitted pipelines without refitting.

---

## 6. Key Literature References (Verified)
1. **KDIGO (2024)**: Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease. *Kidney Int*.
2. **Inker et al. (2021)**: New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race. *NEJM*.
3. **Moons et al. (2025)**: PROBAST+AI: Quality and Risk of Bias Assessment Tool for Prediction Models. *BMJ*.
4. **Collins et al. (2024)**: TRIPOD+AI Statement: Reporting Clinical Prediction Models. *BMJ*.
5. **Lundberg & Lee (2017)**: A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS*.
6. **He et al. (2025)**: Multicenter Retrospective Explainable Early-Stage CKD Prediction. *EClinicalMedicine*.
7. **Chouit et al. (2026)**: Interpretable ML for CKD Prediction with SHAP and LIME. *PLOS ONE*.
8. **Sanmarchi et al. (2023)**: Predict, Diagnose, and Treat CKD with ML: Systematic Review. *J Nephrol*.
9. **Webster et al. (2017)**: Chronic Kidney Disease Seminar. *The Lancet*.
10. **Chen & Guestrin (2016)**: XGBoost: A Scalable Tree Boosting System. *ACM KDD*.
11. **Breiman (2001)**: Random Forests. *Mach Learn*.
12. **Ghorbani et al. (2019)**: Interpretation of Neural Networks Is Fragile. *AAAI*.
