# Comprehensive Research Guide: Explainable Machine Learning for Chronic Kidney Disease Classification Using NHANES 2021–2023 Data

**Project Title**: Explainable Machine Learning for Chronic Kidney Disease Classification Using NHANES 2021–2023 Data: Feature Attribution Stability and Missing-Data Sensitivity  
**Repository**: [https://github.com/Samyuktha8989/CKD-NHANES-XAI](https://github.com/Samyuktha8989/CKD-NHANES-XAI)  
**Reporting Standards**: TRIPOD+AI (2024) and PROBAST+AI (2025) Guidelines  
**Dataset**: Continuous National Health and Nutrition Examination Survey (NHANES) 2021–2023 cycle  

---

## Table of Contents
1. [The Problem: Clinical Context and Machine Learning Pitfalls](#1-the-problem-clinical-context-and-machine-learning-pitfalls)
2. [Dataset Audit, Discovery & Leakage Quarantine](#2-dataset-audit-discovery--leakage-quarantine)
3. [Architecture and Modular Source Scripts](#3-architecture-and-modular-source-scripts)
4. [What is Explainable AI (XAI) and How Was It Implemented?](#4-what-is-explainable-ai-xai-and-how-was-it-implemented)
5. [Validation Protocol and Evaluation Metrics](#5-validation-protocol-and-evaluation-metrics)
6. [Detailed Experimental Results](#6-detailed-experimental-results)
7. [In-Depth Graph Interpretations](#7-in-depth-graph-interpretations)
8. [Clinical & Pathophysiological Interpretation of Risk Factors](#8-clinical--pathophysiological-interpretation-of-risk-factors)
9. [Scientific Limitations and Boundary Conditions](#9-scientific-limitations-and-boundary-conditions)
10. [Reproduction & Execution Reference](#10-reproduction--execution-reference)

---

## 1. The Problem: Clinical Context and Machine Learning Pitfalls

### 1.1 The Clinical Challenge
Chronic kidney disease (CKD) affects over **850 million people worldwide** (>10% of the global population) and represents one of the fastest-growing causes of mortality globally. CKD involves gradual loss of kidney function, culminating in cardiovascular death, accelerated bone-mineral disease, or end-stage renal disease (ESRD) requiring lifelong hemodialysis or kidney transplantation.

Under clinical consensus guidelines established by **KDIGO (Kidney Disease: Improving Global Outcomes 2024)**, CKD is formally defined as:
1. **Reduced Glomerular Filtration**: An estimated Glomerular Filtration Rate ($\text{eGFR}$) $< 60 \text{ mL/min/1.73 m}^2$; and/or
2. **Persistent Kidney Damage**: Evidence of kidney damage, primarily marked by elevated urinary albumin excretion ($\text{Urinary Albumin-to-Creatinine Ratio [UACR]} \ge 30 \text{ mg/g}$), present for $> 3 \text{ months}$.

**The Core Clinical Problem**: In early stages (Stages 1–3a), **CKD is virtually asymptomatic**. Patients do not feel pain, fatigue, or visible symptoms until up to 70–80% of nephron filtration capacity has been irreversibly lost. Consequently, opportunistic risk stratification using routine clinical, demographic, and non-specialized blood chemistry measurements could flag high-risk individuals before overt renal failure develops.

---

### 1.2 The Machine Learning Pitfalls in Existing Medical AI Literature
While hundreds of papers claim machine learning models with "near-perfect" accuracy in CKD, the field suffers from four severe methodological crises:

1. **Tautological Target Leakage (The Fatal Flaw)**:
   - In many published papers, researchers train models using features like `serum_creatinine`, `eGFR`, or `urine_albumin` to predict `ckd_present`.
   - **Why this is fatal**: eGFR is mathematically calculated from serum creatinine using the CKD-EPI formula, and CKD is defined as $\text{eGFR} < 60$. If a model is given `serum_creatinine` or `eGFR` as an input to predict CKD, it is not "learning clinical risk factors"—it is merely reverse-engineering the deterministic algebraic formula of the target!
   - This results in artificial, misleading metrics ($\text{ROC-AUC} > 0.999$) that are completely useless for early pre-diagnostic screening.
2. **The "Black-Box" Opacity Dilemma**:
   - Complex ensemble models (e.g., XGBoost, Random Forest) build complex nonlinear decision boundaries. Without rigorous explainability, clinicians cannot inspect *why* a model flagged a patient, preventing clinical adoption.
3. **Explanation Instability (The Ignored Vulnerability)**:
   - Many studies compute feature importance (such as SHAP) on a single data split. However, post-hoc explanations often fluctuate wildly across different training samples. If a model ranks blood pressure as the #2 feature in one split but drops it to #10 in another, the explanation cannot be clinically trusted.
4. **Uncharacterized Missing-Data Vulnerability**:
   - Real-world electronic health records and epidemiological surveys are plagued by missing laboratory values. Models that achieve high validation accuracy on complete datasets frequently experience catastrophic failure when applied to real-world missing data.

---

## 2. Dataset Audit, Discovery & Leakage Quarantine

### 2.1 The Data Source
We analyzed the **Continuous NHANES 2021–2023 cycle**, an extensive, multi-ethnic, nationally representative health survey conducted by the Centers for Disease Control and Prevention (CDC) National Center for Health Statistics (NCHS).

The raw secondary dataset (`CKD_NHANES_2021_2023.csv`) contains:
- **$11,933$ initial participants**
- **$29$ raw variables**

---

### 2.2 Major Quality Discovery: The "Unknown Stage" Misclassification Trap
During our exploratory data audit ([`src/data_audit.py`](file:///home/user/CKD/src/data_audit.py)), we uncovered a critical flaw in the Kaggle dataset derivation:
- In NHANES, serum creatinine and urinary albumin are **not collected in pediatric participants under 12 years of age** or in unconsented/untested individuals.
- For these individuals ($N = 5,607$), the dataset creator assigned `ckd_stage = 'Unknown'`.
- However, the binary indicator `ckd_present` was generated using the code `(ckd_stage != 'No CKD')`.
- Because `'Unknown' != 'No CKD'`, **5,607 completely untested, healthy children and non-tested individuals were falsely labeled as CKD-positive ($y=1$)!**

**Our Scientific Resolution**:
We filtered out all $5,607$ unmeasured records with `ckd_stage == 'Unknown'`. This produced a verified, clinically robust study cohort of **$N = 6,326$ participants**:
- **No CKD ($y = 0$)**: $3,592$ participants ($56.78\%$)
- **CKD Present ($y = 1$)**: $2,734$ participants ($43.22\%$)

```
+---------------------------------------------------------------------------------------------------+
|                                      NHANES 2021–2023 COHORT                                       |
|                                       Raw: N = 11,933 rows                                        |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
                        [Filter: Untested/Pediatric ckd_stage == 'Unknown']
                                        (Excluded N = 5,607)
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                                      VERIFIED STUDY COHORT                                        |
|                          N = 6,326 (No CKD: 3,592 [56.8%], CKD: 2,734 [43.2%])                     |
+---------------------------------------------------------------------------------------------------+
```

---

### 2.3 Strict Target Leakage Quarantine
To eliminate tautological leakage, we strictly excluded **7 target-defining variables**:
1. `participant_id` (Administrative survey SEQN identifier).
2. `egfr` (Deterministic formula target definition).
3. `serum_creatinine` (Direct input to 2021 CKD-EPI equation).
4. `urine_albumin` (Direct numerator of KDIGO albuminuria criteria).
5. `urine_creatinine` (Direct denominator of KDIGO albuminuria criteria).
6. `albumin_creatinine_ratio` (Direct KDIGO threshold: $\ge 30\text{ mg/g}$).
7. `ckd_stage` (Direct outcome staging label).

**The Final Predictor Matrix**:
We retained **21 non-leaking predictors** (14 continuous, 7 categorical):
- **Demographics & SES (5)**: `age`, `gender`, `ethnicity`, `education_level`, `poverty_income_ratio`.
- **Vitals & Anthropometrics (5)**: `bmi`, `weight_kg`, `height_cm`, `bp_systolic`, `bp_diastolic`.
- **Routine Blood Chemistry (6)**: `blood_urea_nitrogen` (BUN), `albumin_serum`, `phosphorus`, `bicarbonate`, `calcium`, `uric_acid`.
- **Comorbidities & Lifestyle (5)**: `diabetes_diagnosed`, `insulin_use`, `diabetes_pills`, `ever_smoked`, `current_smoker`.

---

## 3. Architecture and Modular Source Scripts

The codebase is organized into modular single-responsibility components with zero hardcoding:

```
CKD/
├── config/
│   └── project.yaml              # Central configuration (paths, hyperparameters, CV, thresholds)
├── src/
│   ├── data_loading.py           # Cohort extraction, schema validation, leakage-safe train/test split
│   ├── data_audit.py             # Descriptive statistics, missingness auditing, distribution profiling
│   ├── leakage_audit.py          # Empirical target-leakage detection and KDIGO compliance checks
│   ├── preprocessing.py          # Leakage-safe ColumnTransformer pipelines
│   ├── models.py                 # LR, Random Forest, XGBoost model instantiators
│   ├── metrics.py                # ROC-AUC, PR-AUC, Calibration (Brier), Sensitivity, Specificity, F1
│   ├── explainability.py         # Exact TreeSHAP and LinearSHAP, Dummy-to-Parent feature aggregator
│   ├── stability_analysis.py     # Pairwise Spearman rho and Top-k Jaccard similarity across 300 fold pairs
│   ├── missingness_sensitivity.py# Controlled MCAR stress-testing engine
│   ├── visualizations.py         # 300 DPI publication figure generation suite
│   ├── reporting.py              # Publication table formatting, artifact manifest, QC report builder
│   └── run_pipeline.py           # Orchestration entry point executing all 7 pipeline stages
├── tests/
│   ├── test_data_loading.py      # Unit tests for loading and cohort filtering
│   ├── test_preprocessing.py     # Unit tests verifying preprocessors fit only on train folds
│   ├── test_models.py            # Unit tests for model predictions and metric computation
│   ├── test_explainability_*.py  # Unit tests for SHAP value computation and shape preservation
│   └── test_integration_smoke.py # End-to-end smoke test verifying pipeline integration
├── manuscript/
│   ├── manuscript.md             # Publication manuscript in GitHub-flavored Markdown
│   ├── manuscript.docx           # Formatted Word manuscript with embedded figures and tables
│   └── generate_docx.py          # Automated script compiling the Word document
├── results/                      # Master output directory containing all 35 validated artifacts
└── README.md                     # Project overview and reproduction instructions
```

### Detailed Script-by-Script Breakdown

| Script | Purpose & Functionality |
| :--- | :--- |
| [`config/project.yaml`](file:///home/user/CKD/config/project.yaml) | **Central Configuration Engine**. Eliminates hardcoding across all scripts. Defines dataset paths, target column, leakage exclusions, 25-fold cross-validation parameters (`n_splits: 5, n_repeats: 5`), fixed random seeds (`42`), model hyperparameters, and SHAP sample limits. |
| [`src/data_loading.py`](file:///home/user/CKD/src/data_loading.py) | **Data Ingestion & Cohort Filter**. Loads raw CSV, audits dimensions, applies the KDIGO filter to drop $5,607$ invalid `'Unknown'` stage records, extracts the verified $N=6,326$ cohort, and executes the feature-target split while quarantining the 7 leakage features. |
| [`src/data_audit.py`](file:///home/user/CKD/src/data_audit.py) | **Exploratory Data Auditor**. Computes missingness percentages for every column, descriptive summary statistics (mean, SD, quartiles, min, max), and target class distributions. Saves tables to `results/data_audit/`. |
| [`src/leakage_audit.py`](file:///home/user/CKD/src/leakage_audit.py) | **Target Leakage Verifier**. Evaluates correlation and mutual information between raw features and the CKD target. Flagged deterministic leakage in `egfr` and `serum_creatinine` and generated `results/data_audit/leakage_audit.csv`. |
| [`src/preprocessing.py`](file:///home/user/CKD/src/preprocessing.py) | **Leakage-Safe Transformer Pipeline**. Implements `sklearn.compose.ColumnTransformer`. Numeric features are processed with `SimpleImputer(strategy='median')` and `StandardScaler`. Categorical features use `SimpleImputer(strategy='most_frequent')` and `OneHotEncoder(handle_unknown='ignore')`. **Critical**: It only exposes `.fit()` for training folds! |
| [`src/models.py`](file:///home/user/CKD/src/models.py) | **Model Architecture Factory**. Instantiates our three contrasting architectures: regularized linear baseline (`LogisticRegression(C=1.0, max_iter=2000)`), bagged nonlinear decision trees (`RandomForestClassifier(n_estimators=300, min_samples_leaf=2)`), and gradient-boosted trees (`XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05)`). |
| [`src/metrics.py`](file:///home/user/CKD/src/metrics.py) | **Comprehensive Metric Calculator**. Implements discrimination metrics ($\text{ROC-AUC}$, $\text{PR-AUC}$), classification metrics at 0.5 decision threshold ($\text{Sensitivity}$, $\text{Specificity}$, $\text{Precision}$, $\text{F1}$, $\text{Accuracy}$, $\text{MCC}$), and probabilistic calibration loss ($\text{Brier Score}$). Computes fold-level and aggregated (Mean ± SD) statistics. |
| [`src/explainability.py`](file:///home/user/CKD/src/explainability.py) | **Explainable AI Engine**. Calculates exact Lundberg TreeSHAP attributions for Random Forest and XGBoost, and LinearSHAP for Logistic Regression. Implements `aggregate_shap_to_parent_features()` to mathematically recombine one-hot dummy levels back into parent clinical features. |
| [`src/stability_analysis.py`](file:///home/user/CKD/src/stability_analysis.py) | **Attribution Stability Quantifier**. Compares SHAP feature importance vectors across all $\binom{25}{2} = 300$ fold combinations. Computes pairwise Spearman rank correlation ($\rho$), Top-5 Jaccard overlap, Top-10 Jaccard overlap, and feature-level rank variability distributions. |
| [`src/missingness_sensitivity.py`](file:///home/user/CKD/src/missingness_sensitivity.py) | **Missingness Stress-Testing Engine**. Injects controlled Missing Completely at Random (MCAR) masking ($10\%$ and $20\%$) into held-out validation inputs across $5$ random seeds. Tracks how discrimination and SHAP rank stability degrade without refitting the training pipeline. |
| [`src/visualizations.py`](file:///home/user/CKD/src/visualizations.py) | **Publication Plotting Suite**. Generates 9 publication-grade 300 DPI figures using matplotlib and seaborn, including ROC/PR curves with confidence bands, calibration reliability curves, global SHAP bar plots, 25×25 stability correlation heatmaps, rank variability boxplots, and missingness degradation trajectories. |
| [`src/reporting.py`](file:///home/user/CKD/src/reporting.py) | **Reporting & Quality Control**. Formats markdown and CSV summary tables, creates `results/metadata/environment_versions.txt`, builds `results/metadata/artifact_manifest.csv`, and writes `project_docs/FINAL_ANALYSIS_QC.md`. |
| [`src/run_pipeline.py`](file:///home/user/CKD/src/run_pipeline.py) | **Central Orchestrator**. Runs the entire 7-stage workflow sequentially, verifying each artifact and outputting timing and summary metrics. |

---

## 4. What is Explainable AI (XAI) and How Was It Implemented?

### 4.1 Theoretical Foundations of SHAP
Traditional machine learning produces opaque probability predictions:
$$\hat{y} = f(\mathbf{x})$$
While this provides risk stratification, it fails to explain *which specific clinical factors* drove an individual prediction.

**Shapley Additive Explanations (SHAP)** (Lundberg & Lee, NeurIPS 2017) grounds feature attribution in cooperative game theory. In this framework:
- The "players" are the clinical features.
- The "game" is predicting the patient's log-odds of CKD.
- The "payout" is the difference between the model's actual prediction and the base expected prediction across the population.

The Shapley value $\phi_j$ for feature $j$ is the unique attribution that satisfies four essential mathematical axioms:
1. **Efficiency**: $\sum_{j=1}^M \phi_j = f(\mathbf{x}) - \mathbb{E}[f(X)]$ (attributions sum to the total deviation from the base rate).
2. **Symmetry**: If two features contribute equally to all possible feature subsets, their attributions are equal.
3. **Dummy / Null**: If a feature contributes nothing to any subset, its attribution is zero.
4. **Additivity**: If a model is an ensemble sum of sub-models, the overall Shapley value is the sum of the sub-models' Shapley values.

---

### 4.2 Algorithmic Implementations in Our Pipeline
In [`src/explainability.py`](file:///home/user/CKD/src/explainability.py), we implemented two distinct SHAP algorithms tailored to model architecture:

1. **Exact TreeSHAP (for XGBoost & Random Forest)**:
   - Evaluates tree ensemble structures recursively in polynomial time $O(TLD^2)$ (where $T$ is the number of trees, $L$ is maximum leaves, and $D$ is tree depth), avoiding exponential combinatorial evaluations.
   - For XGBoost, we interfaced directly with the Lundberg C++ TreeSHAP engine via `booster.predict(dtest, pred_contribs=True)`.
2. **LinearSHAP (for Logistic Regression)**:
   - For linear models $f(\mathbf{x}) = \beta_0 + \sum_{j} \beta_j x_j$, SHAP attributions simplify analytically to:
     $$\phi_j = \beta_j (x_j - \mathbb{E}[x_j])$$
   - Evaluated across an empirical background expectation sample.

---

### 4.3 The Dummy-to-Parent Aggregation Innovation
When machine learning pipelines encode categorical features (such as 6 ethnicity groups or 3 smoking levels) into one-hot dummy columns, naive SHAP packages output separate attributions for each binary column (e.g., `ethnicity_Non-Hispanic White`, `ethnicity_Mexican American`).
- **The Problem**: This artificially splits categorical feature importance across multiple dummy variables, making categorical features look falsely insignificant compared to continuous variables like Age or Blood Pressure!
- **Our Mathematical Solution**: In `aggregate_shap_to_parent_features()`, we implemented a clinically grounded recombination: for patient $i$, the total attribution of parent feature $F$ is the sum of absolute Shapley contributions across all its constituent dummy levels:
  $$\text{Attribution}_i(F) = \sum_{k \in \text{Levels}(F)} |\phi_{i, k}|$$
- Global feature importance is then the mean absolute attribution across all held-out patients:
  $$I(F) = \frac{1}{N} \sum_{i=1}^N \text{Attribution}_i(F)$$

---

## 5. Validation Protocol and Evaluation Metrics

### 5.1 Repeated Stratified 5-Fold Cross-Validation (25 Folds)
A single train/test split (e.g., 80/20) can yield misleading results due to sampling luck. To ensure statistical reliability:
- We used **5 repeats of 5-fold cross-validation** ($5 \times 5 = 25$ total validation folds).
- In each fold, $80\%$ of data ($N \approx 5,060$) was used for training, and $20\%$ ($N \approx 1,266$) was held out for testing.
- **Strict Leakage Isolation**: The preprocessing pipeline was fit exclusively on the training portion of each fold. Test fold samples were never seen during imputer, encoder, or scaler fitting.
- Models were trained, and predictions and SHAP values were generated on the untouched test fold.

---

### 5.2 Mathematical Definitions of Performance Metrics

1. **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**:
   - Measures discrimination across all possible probability classification thresholds.
   - An AUC of $0.5$ represents random guessing; $1.0$ represents perfect separation.
2. **PR-AUC (Precision-Recall Area / Average Precision)**:
   - Area under the precision vs. recall curve. Far more informative than ROC-AUC in biomedical settings because it is sensitive to the true positive class balance ($43.2\%$ prevalence).
3. **Sensitivity (Recall)**:
   $$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   The fraction of true CKD patients correctly detected by the model.
4. **Specificity**:
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
   The fraction of healthy individuals correctly identified as non-CKD.
5. **Precision (Positive Predictive Value - PPV)**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   The fraction of model-flagged patients who truly have CKD.
6. **F1-Score**:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   The harmonic mean of precision and recall.
7. **Matthews Correlation Coefficient (MCC)**:
   $$\text{MCC} = \frac{\text{TP} \times \text{TN} - \text{FP} \times \text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$$
   The gold standard balanced metric for binary classification (ranges from $-1$ to $+1$).
8. **Brier Score (Calibration Loss)**:
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
   Measures the accuracy of probabilistic risk predictions. A score of $0.0$ represents perfect probabilistic calibration.

---

### 5.3 Explanation Stability Metrics Across 300 Fold Pairs
To quantify whether SHAP explanations remain stable when the training sample changes, we compared feature attribution rankings across all $\binom{25}{2} = 300$ pairwise cross-validation fold combinations:

1. **Spearman Rank Correlation ($\rho$)**:
   $$\rho(a, b) = 1 - \frac{6 \sum_{j=1}^M (R_{a, j} - R_{b, j})^2}{M(M^2 - 1)}$$
   Measures monotonic rank agreement of all 21 features between fold $a$ and fold $b$.
2. **Top-$k$ Jaccard Similarity Index ($J_k$)**:
   $$J_k(a, b) = \frac{|S_{k, a} \cap S_{k, b}|}{|S_{k, a} \cup S_{k, b}|}$$
   Measures the set overlap between the top $k$ features identified in fold $a$ vs. fold $b$ ($k \in \{5, 10\}$).

---

## 6. Detailed Experimental Results

### 6.1 Model Performance Across 25 Folds (RQ1)
*Source Table: [`results/tables/model_comparison_table.csv`](file:///home/user/CKD/results/tables/model_comparison_table.csv)*

| Metric | Logistic Regression | Random Forest | XGBoost | Best Performing |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | $0.8441 \pm 0.0123$ | $0.8406 \pm 0.0123$ | **$0.8456 \pm 0.0129$** | **XGBoost** |
| **PR-AUC** | $0.8242 \pm 0.0143$ | $0.8133 \pm 0.0160$ | **$0.8246 \pm 0.0149$** | **XGBoost** |
| **Sensitivity** | **$0.7303 \pm 0.0154$** | $0.7187 \pm 0.0144$ | $0.7025 \pm 0.0161$ | **Logistic Regression** |
| **Specificity** | $0.8142 \pm 0.0171$ | $0.8130 \pm 0.0184$ | **$0.8268 \pm 0.0158$** | **XGBoost** |
| **Precision** | $0.7498 \pm 0.0181$ | $0.7456 \pm 0.0195$ | **$0.7556 \pm 0.0186$** | **XGBoost** |
| **F1-Score** | **$0.7398 \pm 0.0137$** | $0.7317 \pm 0.0133$ | $0.7280 \pm 0.0148$ | **Logistic Regression** |
| **Accuracy** | **$0.7779 \pm 0.0124$** | $0.7722 \pm 0.0124$ | $0.7731 \pm 0.0128$ | **Logistic Regression** |
| **MCC** | **$0.5464 \pm 0.0250$** | $0.5344 \pm 0.0250$ | $0.5350 \pm 0.0263$ | **Logistic Regression** |
| **Brier Score** | **$0.1559 \pm 0.0063$** | $0.1600 \pm 0.0057$ | **$0.1559 \pm 0.0069$** | **LR & XGBoost (Tie)** |

#### Key Insights from RQ1:
- All three architectures achieve strong, comparable discrimination ($\text{ROC-AUC} \approx 0.841–0.846$), proving that systemic routine risk factors carry strong predictive power for concurrent CKD even without direct filtration markers.
- **The Linear Baseline Discovery**: Regularized Logistic Regression matches XGBoost ($\Delta \text{AUC} < 0.002$). This proves that the predictive signal is driven by genuine monotonic physiological relationships, and confirms that complex models did not overfit.

---

### 6.2 Global SHAP Feature Importance Rankings (RQ2)
*Source Tables: [`results/explanations/`](file:///home/user/CKD/results/explanations/)*

| Rank | Feature | XGBoost \|SHAP\| | Random Forest \|SHAP\| | Logistic Reg \|SHAP\| | Clinical Relevance |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **`age`** | **0.1704** | **0.1416** | **0.1406** | Progressive loss of nephrons with advancing age. |
| **2** | **`blood_urea_nitrogen`** | **0.1245** | **0.1062** | **0.0898** | Nitrogenous waste byproduct cleared by renal filtration. |
| **3** | **`bp_systolic`** | **0.0682** | **0.0714** | **0.0645** | Glomerular hypertension & microvascular damage. |
| **4** | **`uric_acid`** | **0.0573** | **0.0621** | **0.0589** | Endothelial dysfunction & renal vasoconstriction. |
| **5** | **`diabetes_diagnosed`** | **0.0512** | **0.0543** | **0.0562** | Diabetic glomerulosclerosis (leading CKD cause). |
| **6** | `bmi` | 0.0384 | 0.0412 | 0.0321 | Obesity-related hyperfiltration injury. |
| **7** | `albumin_serum` | 0.0365 | 0.0398 | 0.0345 | Systemic inflammation & nutritional marker. |
| **8** | `poverty_income_ratio` | 0.0312 | 0.0341 | 0.0289 | Social determinant of healthcare access. |
| **9** | `bicarbonate` | 0.0298 | 0.0315 | 0.0274 | Acid-base balance; subclinical metabolic acidosis. |
| **10** | `phosphorus` | 0.0245 | 0.0268 | 0.0231 | Mineral metabolism in impaired clearance. |

---

### 6.3 SHAP Attribution Stability Across 300 Fold Pairs (RQ3)
*Source Table: [`results/tables/stability_summary_table.csv`](file:///home/user/CKD/results/tables/stability_summary_table.csv)*

| Model | Comparisons | Spearman $\rho$ (Mean ± SD) | Spearman Median (IQR) | Top-5 Jaccard | Top-10 Jaccard |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 300 | **$0.9652 \pm 0.0148$** | **0.9662 (0.0208)** | **$1.0000 \pm 0.0000$** | **$0.9577 \pm 0.0781$** |
| **XGBoost** | 300 | $0.9414 \pm 0.0214$ | 0.9455 (0.0312) | **$1.0000 \pm 0.0000$** | $0.7997 \pm 0.1087$ |
| **Logistic Regression** | 300 | $0.8972 \pm 0.0553$ | 0.9117 (0.0740) | $0.9267 \pm 0.1383$ | $0.8162 \pm 0.1022$ |

> **Major Finding**: Both Random Forest and XGBoost achieved **$100\%$ perfect Top-5 feature selection stability ($1.0000 \pm 0.0000$) across all 300 comparisons**. The top 5 clinical risk factors never changed regardless of how the training data were partitioned.

---

### 6.4 Controlled Missing-Data Stress Testing (RQ4)
*Source Table: [`results/tables/missingness_summary_table.csv`](file:///home/user/CKD/results/tables/missingness_summary_table.csv)*

| Model | Masking Stress | ROC-AUC (Mean ± SD) | F1-Score | Accuracy | SHAP Rank Stability ($\rho$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Baseline (0%) | 0.8482 | 0.7422 | 0.7844 | 1.0000 |
| | **10% MCAR** | $0.8366 \pm 0.0023$ | $0.7203 \pm 0.0057$ | $0.7671 \pm 0.0054$ | $0.9403 \pm 0.0037$ |
| | **20% MCAR** | $0.8207 \pm 0.0045$ | $0.7003 \pm 0.0113$ | $0.7501 \pm 0.0108$ | $0.9314 \pm 0.0077$ |
| **Random Forest** | Baseline (0%) | 0.8475 | 0.7342 | 0.7804 | 1.0000 |
| | **10% MCAR** | $0.8370 \pm 0.0021$ | $0.7133 \pm 0.0061$ | $0.7679 \pm 0.0041$ | **$0.9748 \pm 0.0052$** |
| | **20% MCAR** | $0.8227 \pm 0.0028$ | $0.6870 \pm 0.0056$ | $0.7528 \pm 0.0054$ | **$0.9727 \pm 0.0049$** |
| **XGBoost** | Baseline (0%) | 0.8514 | 0.7264 | 0.7757 | 1.0000 |
| | **10% MCAR** | $0.8385 \pm 0.0029$ | $0.7052 \pm 0.0059$ | $0.7626 \pm 0.0045$ | $0.9590 \pm 0.0103$ |
| | **20% MCAR** | $0.8212 \pm 0.0039$ | $0.6756 \pm 0.0066$ | $0.7439 \pm 0.0075$ | $0.9501 \pm 0.0111$ |

---

## 7. In-Depth Graph Interpretations

All 9 figures in [`results/figures/`](file:///home/user/CKD/results/figures/) were generated at 300 DPI:

### 1. [`target_distribution.png`](file:///home/user/CKD/results/figures/target_distribution.png) — Cohort Derivation & Staging Breakdown
- **Left Panel**: Shows the class balance of the analyzed cohort ($3,592$ No CKD [56.8%], $2,734$ CKD [43.2%]).
- **Right Panel**: Illustrates the detailed KDIGO 2024 clinical stage breakdown (Stage 1, 2, 3a, 3b, 4, 5).
- **Interpretation**: Confirms that filtering out the $5,607$ untested pediatric records produced a well-balanced cohort without severe class imbalance, eliminating the need for synthetic oversampling (SMOTE).

---

### 2. [`feature_missingness.png`](file:///home/user/CKD/results/figures/feature_missingness.png) — Feature Missingness Profile
- **What it shows**: Horizontal bar plot displaying the exact missingness percentage for every predictor.
- **Interpretation**: Core metabolic labs (Age, BUN, Uric Acid, Serum Albumin) have near-zero missingness ($<0.1\%$). Highest missingness occurs in conditional lifestyle questions (`insulin_use`: $87.9\%$, `diabetes_pills`: $74.0\%$, `current_smoker`: $64.6\%$). This justified using training-fold median/mode imputation rather than dropping rows.

---

### 3. [`roc_pr_curves.png`](file:///home/user/CKD/results/figures/roc_pr_curves.png) — ROC and Precision-Recall Curves
- **Left Panel (ROC Curves)**: Compares the True Positive Rate vs. False Positive Rate across all 25 folds with shaded $\pm 1$ SD confidence bands. All three curves closely overlap ($\text{AUC} \approx 0.841–0.846$).
- **Right Panel (Precision-Recall Curves)**: Precision vs. Recall curves. The horizontal dashed baseline represents random chance ($0.432$). All models achieve $\text{PR-AUC} > 0.813$, maintaining high precision even at high recall levels.
- **Interpretation**: Proves that non-leaking predictors provide high, stable discrimination across repeated resampling.

---

### 4. [`model_comparison.png`](file:///home/user/CKD/results/figures/model_comparison.png) — Multi-Metric Performance Comparison
- **What it shows**: Grouped bar charts comparing ROC-AUC, PR-AUC, F1-Score, Accuracy, Sensitivity, and Specificity across Logistic Regression, Random Forest, and XGBoost.
- **Interpretation**: Highlights that while XGBoost has slightly higher specificity ($0.827$), Logistic Regression achieves higher sensitivity ($0.730$) and F1-score ($0.740$), demonstrating that simple linear models remain competitive with modern gradient boosting for tabular clinical data.

---

### 5. [`calibration_plots.png`](file:///home/user/CKD/results/figures/calibration_plots.png) — Probabilistic Reliability Curves
- **What it shows**: Observed fraction of CKD positives plotted against predicted risk probability across 10 risk deciles. The diagonal dashed line represents perfect calibration.
- **Interpretation**: All three models closely track the diagonal. Models do not suffer from probability distortion or overconfidence, confirming that predicted probabilities can be directly interpreted as clinical risk estimates (Brier score $= 0.156$).

---

### 6. [`global_feature_importance.png`](file:///home/user/CKD/results/figures/global_feature_importance.png) — Global SHAP Feature Rankings
- **What it shows**: Horizontal bar charts displaying mean absolute SHAP values for each parent clinical feature across Logistic Regression, Random Forest, and XGBoost.
- **Interpretation**: Visually proves the consensus: **Age is #1**, followed by **BUN (#2)**, **Systolic BP (#3)**, **Uric Acid (#4)**, and **Diagnosed Diabetes (#5)**. These top 5 features dominate model predictions across all three architectures.

---

### 7. [`shap_stability_heatmap.png`](file:///home/user/CKD/results/figures/shap_stability_heatmap.png) — 25×25 Fold Stability Heatmaps
- **What it shows**: $25 \times 25$ correlation matrix heatmaps showing the pairwise Spearman rank correlation ($\rho$) between every pair of cross-validation folds for each model.
- **Interpretation**: Random Forest and XGBoost heatmaps are uniformly dark/saturated ($\rho > 0.94–0.96$ across all cells), visually demonstrating that tree-based feature attributions do not fluctuate between folds. Logistic Regression shows slightly more variability ($\rho \approx 0.90$), reflecting collinearity between blood pressure and glycemic markers.

---

### 8. [`feature_rank_variability.png`](file:///home/user/CKD/results/figures/feature_rank_variability.png) — Rank Stability Boxplots
- **What it shows**: Boxplots showing the distribution of ranks assigned to each feature across all 25 cross-validation folds.
- **Interpretation**: Features like `age` (Rank 1, zero variance), `blood_urea_nitrogen` (Rank 2, zero variance), and `bp_systolic` have narrow boxplots with near-zero spread. Rank fluctuation only occurs among lower-importance features (e.g., education level vs poverty ratio), proving that clinical conclusions about the top risk factors are completely stable.

---

### 9. [`missingness_sensitivity.png`](file:///home/user/CKD/results/figures/missingness_sensitivity.png) — Degradation Trajectories Under Missing Data
- **Left Panel (ROC-AUC Degradation)**: Plots ROC-AUC across 0%, 10%, and 20% synthetic MCAR missingness. Shows a smooth, linear degradation of only $\approx 0.012$ points at 10% and $\approx 0.026$ points at 20%.
- **Right Panel (SHAP Rank Stability)**: Shows the Spearman rank correlation of feature importance relative to baseline. Random Forest maintains $\rho > 0.97$, XGBoost maintains $\rho > 0.95$, and Logistic Regression maintains $\rho > 0.93$.
- **Interpretation**: Confirms that our models exhibit **graceful degradation** rather than catastrophic failure when clinical data are missing, and that explanations remain trustworthy even in data-scarce settings.

---

## 8. Clinical & Pathophysiological Interpretation of Risk Factors

The global SHAP analysis revealed a medically coherent hierarchy of predictors:

### 1. Chronological Age (#1 Predictor, Mean |SHAP| ~0.14–0.17)
- **Pathophysiology**: Normal aging leads to progressive loss of functional nephrons (approx. 10% loss per decade after age 40), glomerulosclerosis, tubulointerstitial fibrosis, and arterial stiffening (Denic et al., 2016). In our cohort, mean age was **$60.4$ years in CKD patients** versus **$39.6$ years in non-CKD controls**.

### 2. Blood Urea Nitrogen (BUN) (#2 Predictor, Mean |SHAP| ~0.09–0.12)
- **Pathophysiology**: Urea is the primary nitrogenous end-product of protein catabolism, cleared almost exclusively by glomerular filtration. Elevated BUN ($17.8\text{ mg/dL}$ in CKD vs $12.3\text{ mg/dL}$ in non-CKD) provides a strong non-creatinine proxy for declining renal clearance capacity.

### 3. Systolic Blood Pressure (#3 Predictor, Mean |SHAP| ~0.06–0.07)
- **Pathophysiology**: Systemic hypertension transmits elevated mechanical barotrauma directly to glomerular capillary beds, causing endothelial shear stress, glomerular hyperfiltration injury, and progressive hypertensive nephrosclerosis (Chobanian et al., 2003). Mean systolic BP was **$127.8\text{ mmHg}$ in CKD** vs **$115.3\text{ mmHg}$ in non-CKD**.

### 4. Serum Uric Acid (#4 Predictor, Mean |SHAP| ~0.05–0.06)
- **Pathophysiology**: Hyperuricemia induces afferent arteriolopathy, activates the intrarenal renin-angiotensin-aldosterone system (RAAS), inhibits endothelial nitric oxide release, and directly drives renal tubular epithelial oxidative injury (Johnson et al., 2018). Mean uric acid was elevated at **$5.56\text{ mg/dL}$ in CKD** vs **$4.81\text{ mg/dL}$ in non-CKD**.

### 5. Diagnosed Diabetes Status (#5 Predictor, Mean |SHAP| ~0.05–0.06)
- **Pathophysiology**: Diabetic kidney disease (DKD) is the single leading cause of end-stage renal disease worldwide. Chronic hyperglycemia drives non-enzymatic protein glycation, mesangial expansion, podocyte effacement, and basement membrane thickening. Diagnosed diabetes prevalence was **$23.5\%$ in the CKD cohort** vs **$5.1\%$ in non-CKD controls**.

---

## 9. Scientific Limitations and Boundary Conditions

To maintain academic integrity and prevent clinical misinterpretation:

1. **Cross-Sectional Classification vs. Longitudinal Inception**:
   - NHANES is an epidemiological cross-sectional survey with single-timepoint laboratory measurements.
   - KDIGO criteria formally define CKD as functional abnormalities persisting for $>3$ months. Single-timepoint measurements cannot distinguish acute kidney injury (AKI) from chronic disease.
   - **Conclusion**: Our models classify **concurrent CKD risk-factor status**, not prospective future disease onset.
2. **Pre-Diagnostic Screening Scope**:
   - Excluding direct diagnostic criteria (eGFR, serum creatinine, UACR) is essential to avoid tautological leakage, but naturally caps maximum discriminative ceiling at $\text{ROC-AUC} \approx 0.85$.
   - These models are designed for **opportunistic risk screening in primary care** when full nephrology lab panels have not yet been ordered.
3. **Non-Diagnostic Status**:
   - These models are algorithmic risk classifiers and must **never** be used as automated clinical diagnostic devices to replace laboratory testing.

---

## 10. Reproduction & Execution Reference

The entire pipeline is open-source and reproducible:

```bash
# 1. Verify complete test suite (12/12 tests passing)
PYTHONPATH=deps:. pytest tests/ -v

# 2. Run complete end-to-end analysis pipeline (7 stages)
PYTHONPATH=deps:. python3 src/run_pipeline.py

# 3. Compile Word document manuscript
PYTHONPATH=deps:. python3 manuscript/generate_docx.py
```

### Git Repository Status
The complete project—including source code, 35 results artifacts, tests, configurations, and the publication manuscript—is published and available at:
👉 **[https://github.com/Samyuktha8989/CKD-NHANES-XAI](https://github.com/Samyuktha8989/CKD-NHANES-XAI)**
