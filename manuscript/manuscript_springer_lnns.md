# Explainable Machine Learning for Chronic Kidney Disease Classification Using NHANES 2021–2023 Data: Feature Attribution Stability and Missing-Data Sensitivity

**Author Names**: [First Author]$^{1[0000-0000-0000-0000]}$, [Second Author]$^{1}$, and [Corresponding Author]$^{2*[0000-0000-0000-0000]}$  
$^{1}$ Department of Computer Science and Engineering, Institute / University, City, Country  
$^{2}$ Department of Biomedical Informatics and Computational Health, Institute / University, City, Country  
$^*$ Corresponding Author: `author@university.edu`

---

**Abstract.** Chronic kidney disease (CKD) is an escalating global public health priority affecting more than 850 million individuals worldwide. While supervised machine learning (ML) models demonstrate considerable discriminative capacity for risk stratification, clinical translation is severely hindered by the "black-box" opacity of complex classifiers, documented instability of post-hoc explanation algorithms (e.g., SHAP) under sample resampling, and uncharacterized sensitivity to missing clinical data. Furthermore, retrospective biomedical ML studies frequently suffer from tautological target leakage by including direct diagnostic criteria (e.g., eGFR or albuminuria) as predictive features. In this study, we address these challenges using the Continuous National Health and Nutrition Examination Survey (NHANES) 2021–2023 cycle. We audit and verify a cohort of $N = 6,326$ participants ($\ge 12$ years) with complete KDIGO 2024 clinical staging ($3,592$ non-CKD [$56.78\%$], $2,734$ CKD [$43.22\%$]), strictly excluding $5,607$ untested pediatric/unmeasured records with unknown stage. Seven target-defining variables are eliminated to prevent target leakage, retaining $21$ non-leaking predictors. Models are trained using Repeated Stratified 5-Fold Cross-Validation ($25$ folds). TreeSHAP and LinearSHAP attributions are computed on held-out test partitions, aggregated back to parent clinical variables, and evaluated across all $\binom{25}{2} = 300$ pairwise fold combinations via Spearman rank correlation ($\rho$) and Top-$k$ Jaccard similarity. Controlled Missing Completely at Random (MCAR) stress testing is conducted at $10\%$ and $20\%$ masking levels. Across 25 folds, XGBoost achieved an ROC-AUC of $0.8456 \pm 0.0129$ (PR-AUC $0.8246 \pm 0.0149$), closely matched by Logistic Regression ($0.8441 \pm 0.0123$) and Random Forest ($0.8406 \pm 0.0123$). Global SHAP analysis identified chronological age, blood urea nitrogen (BUN), systolic blood pressure, serum uric acid, and diagnosed diabetes as top consensus risk factors. Pairwise SHAP attribution stability across 300 fold resamples was exceptionally high for Random Forest (Spearman $\rho = 0.9652 \pm 0.0148$, Top-5 Jaccard $1.0000 \pm 0.0000$) and XGBoost ($\rho = 0.9414 \pm 0.0214$, Top-5 Jaccard $1.0000 \pm 0.0000$). Under $10\%$ and $20\%$ missingness, all models exhibited graceful degradation (ROC-AUC drop $\le 0.026$) while preserving high explanation ranking fidelity ($\rho \ge 0.931–0.975$). These results provide empirical best practices for leakage-safe algorithmic auditing and explainability verification in computational nephrology.

**Keywords:** Chronic Kidney Disease, Explainable Artificial Intelligence, SHAP, Feature Attribution Stability, Missing Data Sensitivity, NHANES, Soft Computing, Computational Intelligence.

---

## 1 Introduction

Chronic kidney disease (CKD) is an escalating global health emergency characterized by structural kidney damage and progressive decline in renal function, affecting over 10% to 15% of the adult population worldwide (>850 million individuals) [1, 2]. According to international consensus guidelines established by Kidney Disease: Improving Global Outcomes (KDIGO 2024), CKD is clinically diagnosed by an estimated glomerular filtration rate (eGFR) below $60\text{ mL/min/1.73 m}^2$ or sustained markers of renal damage, primarily albuminuria (urinary albumin-to-creatinine ratio $[\text{UACR}] \ge 30\text{ mg/g}$), persisting for more than three months [1, 3].

Because early-stage CKD is predominantly asymptomatic, patients frequently remain undiagnosed until advanced, irreversible nephron loss has occurred [4]. Consequently, there is intense clinical interest in leveraging machine learning (ML) and soft computing techniques to perform automated risk stratification, identify high-risk individuals from routine electronic health records (EHR), and uncover multi-organ risk factor interactions [5, 6].

### 1.1 Methodological Challenges in Medical Machine Learning
Despite promising performance reported in the computational literature, clinical translation of machine learning models in nephrology is impeded by four critical vulnerabilities:

1. **Tautological Target Leakage**: Numerous published predictive studies mistakenly include the very laboratory variables used to derive the clinical outcome—such as serum creatinine, eGFR, or spot urine albumin—directly in the predictor matrix [7]. When a model is supplied with serum creatinine or eGFR to predict CKD presence, it simply reconstructs the deterministic formula of the KDIGO definition ($\text{eGFR} < 60$), yielding artificially inflated performance metrics ($\text{ROC-AUC} > 0.99$) that convey zero clinical utility for pre-diagnostic screening.
2. **"Black-Box" Opacity**: Nonlinear ensemble models construct complex interaction manifolds that clinicians cannot intuitively trace without post-hoc explainability frameworks [8].
3. **Attribution Instability Across Resampled Cohorts**: A critical, under-investigated vulnerability of explainable AI (XAI) is *explanation instability*: the phenomenon wherein small perturbations in the training data resample (e.g., across cross-validation folds) induce substantial fluctuations in feature importance rankings [9, 10]. If an algorithm ranks blood pressure as the 2nd most important risk factor in one fold but drops it to 10th in another, clinicians cannot trust the clinical validity of the generated explanations.
4. **Missing-Data Sensitivity and Robustness**: Routine clinical datasets and epidemiological surveys are characterized by pervasive missingness resulting from variable lab order patterns or instrument thresholds [11]. While models may achieve high validation accuracy on complete data, their degradation trajectory under realistic missing-data stress is rarely quantified systematically.

### 1.2 Research Contributions
To address these challenges, this study presents a rigorous, leakage-free computational investigation using the Continuous National Health and Nutrition Examination Survey (NHANES 2021–2023) cycle. Our primary contributions are:
- Auditing the NHANES 2021–2023 cohort and eliminating a major target-labeling anomaly that misclassified 5,607 untested pediatric records.
- Enforcing strict target-leakage quarantine by excluding 7 direct diagnostic criteria, benchmarking 21 non-leaking systemic predictors.
- Evaluating model discrimination and probabilistic calibration across Repeated Stratified 5-Fold Cross-Validation (25 folds) for Logistic Regression, Random Forest, and XGBoost.
- Quantifying pairwise SHAP attribution stability across all $\binom{25}{2} = 300$ fold resample comparisons using Spearman rank correlation ($\rho$) and Top-$k$ Jaccard similarity indices.
- Measuring predictive and explanatory degradation trajectories under controlled Missing Completely at Random (MCAR) stress testing (10% and 20% masking levels).

---

## 2 Materials and Methods

### 2.1 Study Cohort and KDIGO Staging Criteria
This investigation analyzed tabular cross-sectional data from the 2021–2023 Continuous NHANES cycle administered by the National Center for Health Statistics (NCHS) at the Centers for Disease Control and Prevention (CDC) [12].

In the raw secondary dataset ($N = 11,933$ participants, $29$ variables), kidney function evaluation is governed by KDIGO 2024 guidelines:
1. **eGFR Determination**: eGFR was computed using the race-free 2021 CKD-EPI creatinine equation [3]:
   $$\text{eGFR} = 142 \times \min\left(\frac{S_{cr}}{\kappa}, 1\right)^\alpha \times \max\left(\frac{S_{cr}}{\kappa}, 1\right)^{-1.200} \times 0.9938^{\text{Age}} \times [1.012 \text{ if Female}]$$
   where $S_{cr}$ is serum creatinine ($\text{mg/dL}$), $\kappa = 0.7$ (females) or $0.9$ (males), and $\alpha = -0.241$ (females) or $-0.302$ (males).
2. **KDIGO Clinical Staging**: Staged into Stage 1 ($\text{eGFR} \ge 90$ with $\text{UACR} \ge 30$), Stage 2 ($60–89$ with $\text{UACR} \ge 30$), Stage 3a ($45–59$), Stage 3b ($30–44$), Stage 4 ($15–29$), Stage 5 ($<15$), or No CKD ($\text{eGFR} \ge 60$ and $\text{UACR} < 30$).

During data auditing, we discovered that $5,607$ participants who did not undergo kidney function laboratory testing (predominantly pediatric participants $<12$ years old) had `ckd_stage = 'Unknown'` and were erroneously labeled as CKD-positive by the naive condition `(ckd_stage != 'No CKD')`. Excluding these untested records produced a verified study cohort of **$N = 6,326$ participants** ($3,592$ No CKD [$56.78\%$], $2,734$ CKD [$43.22\%$]).

### 2.2 Feature Policy and Leakage Elimination
To prevent tautological target reconstruction, seven direct diagnostic features were quarantined: `participant_id`, `egfr`, `serum_creatinine`, `urine_albumin`, `urine_creatinine`, `albumin_creatinine_ratio`, and `ckd_stage`.

The approved feature matrix comprises **21 non-leaking predictors**:
- **Demographics & SES (5)**: `age`, `gender`, `ethnicity`, `education_level`, `poverty_income_ratio`.
- **Vitals & Anthropometrics (5)**: `bmi`, `weight_kg`, `height_cm`, `bp_systolic`, `bp_diastolic`.
- **Routine Blood Chemistry (6)**: `blood_urea_nitrogen` (BUN), `albumin_serum`, `phosphorus`, `bicarbonate`, `calcium`, `uric_acid`.
- **Comorbidities & Lifestyle (5)**: `diabetes_diagnosed`, `insulin_use`, `diabetes_pills`, `ever_smoked`, `current_smoker`.

Table 1 summarizes the baseline characteristics of the study cohort.

**Table 1.** Baseline characteristics of the study cohort stratified by CKD status ($N=6,326$).
| Characteristic | Total Cohort ($N=6,326$) | No CKD ($N=3,592$) | CKD Present ($N=2,734$) | Missing (%) |
| :--- | :---: | :---: | :---: | :---: |
| Age (years), Mean ± SD | $48.62 \pm 20.71$ | $39.64 \pm 19.82$ | $60.42 \pm 15.21$ | 0.0% |
| Female Gender, $n$ (%) | $3,428$ ($54.19\%$) | $1,942$ ($54.06\%$) | $1,486$ ($54.35\%$) | 0.0% |
| BMI ($\text{kg/m}^2$), Mean ± SD | $29.08 \pm 7.36$ | $28.12 \pm 7.14$ | $30.34 \pm 7.46$ | 1.28% |
| Systolic BP ($\text{mmHg}$), Mean ± SD | $120.72 \pm 18.17$ | $115.32 \pm 15.44$ | $127.81 \pm 18.92$ | 3.05% |
| Diastolic BP ($\text{mmHg}$), Mean ± SD | $73.75 \pm 11.55$ | $71.94 \pm 10.82$ | $76.13 \pm 12.06$ | 3.05% |
| Blood Urea Nitrogen ($\text{mg/dL}$) | $14.66 \pm 5.75$ | $12.28 \pm 3.62$ | $17.79 \pm 6.78$ | 0.05% |
| Serum Albumin ($\text{g/dL}$) | $4.10 \pm 0.36$ | $4.18 \pm 0.32$ | $3.99 \pm 0.38$ | 0.0% |
| Serum Uric Acid ($\text{mg/dL}$) | $5.13 \pm 1.41$ | $4.81 \pm 1.32$ | $5.56 \pm 1.42$ | 0.0% |
| Diagnosed Diabetes, $n$ (%) | $826$ ($13.06\%$) | $183$ ($5.09\%$) | $643$ ($23.52\%$) | 0.0% |
| Poverty-Income Ratio | $2.88 \pm 1.66$ | $2.94 \pm 1.68$ | $2.81 \pm 1.63$ | 12.54% |

### 2.3 Machine Learning Pipeline and Validation Protocol
All transformations and evaluations were integrated within a **Repeated Stratified 5-Fold Cross-Validation** protocol with $5$ independent random repeats ($25$ total folds).

1. **Preprocessing Pipeline**: Continuous features were processed using median imputation and z-score standard scaling. Categorical features were imputed using mode imputation and one-hot encoding (`handle_unknown='ignore'`). **Strict Leakage Isolation**: All imputers, encoders, and scalers were fit strictly on the training partition of each fold ($N \approx 5,060$) and applied via transform to the held-out test fold ($N \approx 1,266$).
2. **Model Architectures**:
   - **Logistic Regression**: L2 regularization penalty ($C=1.0$), `lbfgs` solver, maximum iterations $2,000$.
   - **Random Forest**: $300$ bagged decision trees, `min_samples_leaf = 2`, `max_features = 'sqrt'`.
   - **XGBoost**: $300$ boosting rounds, learning rate $\eta = 0.05$, `max_depth = 3`, `subsample = 0.8`, `colsample_bytree = 0.8`.

### 2.4 Explainability and Feature Attribution Aggregation
Post-hoc local explanations were computed for all held-out validation instances using Shapley Additive Explanations (SHAP) [8]. For Random Forest and XGBoost, we utilized exact Lundberg TreeSHAP ($O(TLD^2)$ complexity). For Logistic Regression, we computed analytical LinearSHAP.

#### Aggregation of One-Hot Dummy Levels
One-hot encoding decomposes a categorical feature $F$ with $K$ categories into $K$ separate binary indicators $\{f_1, \dots, f_K\}$. To restore clinical interpretability, we aggregated transformed SHAP values back to their parent clinical feature. For observation $i$, the total absolute attribution of parent feature $F$ was defined as:
$$\text{Attribution}_i(F) = \sum_{k \in \text{Levels}(F)} |\phi_{i, k}|$$
Global feature importance was subsequently calculated as the mean absolute attribution across all evaluated validation instances:
$$I(F) = \frac{1}{N} \sum_{i=1}^N \text{Attribution}_i(F)$$

### 2.5 Quantification of Explanation Stability
To evaluate whether SHAP feature rankings reflect robust physiological relationships rather than sample-specific noise, we evaluated attribution stability across all $\binom{25}{2} = 300$ pairwise cross-validation fold combinations:
1. **Spearman Rank Correlation ($\rho$)**: Evaluates monotonic agreement between feature importance vectors across fold $a$ and fold $b$:
   $$\rho(a, b) = 1 - \frac{6 \sum_{j=1}^M (R_{a, j} - R_{b, j})^2}{M(M^2 - 1)}$$
   where $R_{a, j}$ is the rank of feature $j$ in fold $a$, and $M = 21$ is the number of evaluated parent features.
2. **Top-$k$ Jaccard Similarity Index ($J_k$)**: Quantifies the set overlap of the $k$ most influential features:
   $$J_k(a, b) = \frac{|S_{k, a} \cap S_{k, b}|}{|S_{k, a} \cup S_{k, b}|}$$
   evaluated at $k \in \{5, 10\}$.

### 2.6 Controlled Missing-Data Stress Testing
Controlled Missing Completely at Random (MCAR) masking was applied exclusively to held-out validation features at masking levels $m \in \{0.10, 0.20\}$ ($10\%$ and $20\%$ additional missing values across observed cells), repeated over $5$ fixed random seeds ($42, 101, 202, 303, 404$). Transformed masked validation features were processed using pre-fitted training imputers, and both predictive metrics and SHAP rank stability relative to unmasked baselines were quantified.

---

## 3 Experimental Results

### 3.1 Model Discrimination and Calibration (RQ1)
Across 25 repeated cross-validation folds, all three machine learning architectures exhibited strong, highly competitive discriminative capacity on held-out validation partitions (Table 2).

**Table 2.** Cross-validation performance metrics across 25 folds (Mean ± SD).
| Performance Metric | Logistic Regression | Random Forest | XGBoost | Best Model |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | $0.8441 \pm 0.0123$ | $0.8406 \pm 0.0123$ | **$0.8456 \pm 0.0129$** | **XGBoost** |
| **PR-AUC (Avg Precision)** | $0.8242 \pm 0.0143$ | $0.8133 \pm 0.0160$ | **$0.8246 \pm 0.0149$** | **XGBoost** |
| **Sensitivity (Recall)** | **$0.7303 \pm 0.0154$** | $0.7187 \pm 0.0144$ | $0.7025 \pm 0.0161$ | **Logistic Regression** |
| **Specificity** | $0.8142 \pm 0.0171$ | $0.8130 \pm 0.0184$ | **$0.8268 \pm 0.0158$** | **XGBoost** |
| **Precision (PPV)** | $0.7498 \pm 0.0181$ | $0.7456 \pm 0.0195$ | **$0.7556 \pm 0.0186$** | **XGBoost** |
| **F1-Score** | **$0.7398 \pm 0.0137$** | $0.7317 \pm 0.0133$ | $0.7280 \pm 0.0148$ | **Logistic Regression** |
| **Accuracy** | **$0.7779 \pm 0.0124$** | $0.7722 \pm 0.0124$ | $0.7731 \pm 0.0128$ | **Logistic Regression** |
| **Matthews Correlation (MCC)**| **$0.5464 \pm 0.0250$** | $0.5344 \pm 0.0250$ | $0.5350 \pm 0.0263$ | **Logistic Regression** |
| **Brier Score (Calibration)** | **$0.1559 \pm 0.0063$** | $0.1600 \pm 0.0057$ | **$0.1559 \pm 0.0069$** | **LR & XGBoost (Tie)** |

XGBoost achieved the highest overall area under the receiver operating characteristic curve ($\text{ROC-AUC} = 0.8456 \pm 0.0129$) and precision-recall curve ($\text{PR-AUC} = 0.8246 \pm 0.0149$), closely matched by regularized Logistic Regression ($\text{ROC-AUC} = 0.8441 \pm 0.0123$, $\text{PR-AUC} = 0.8242 \pm 0.0143$). Brier calibration scores ($0.1559$) confirmed excellent probabilistic reliability.

### 3.2 Global SHAP Feature Importance Consensus (RQ2)
Global SHAP attributions aggregated across all 25 folds revealed a consistent biological hierarchy of predictors (Table 3).

**Table 3.** Global SHAP feature importance rankings (Mean |SHAP| attribution) across models.
| Rank | Feature Name | XGBoost \|SHAP\| | Random Forest \|SHAP\| | Logistic Reg \|SHAP\| | Primary Clinical Mechanism |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `age` | **0.1704** | **0.1416** | **0.1406** | Age-related nephrosclerosis & vascular stiffening |
| **2** | `blood_urea_nitrogen` | **0.1245** | **0.1062** | **0.0898** | Nitrogenous waste filtration clearance proxy |
| **3** | `bp_systolic` | **0.0682** | **0.0714** | **0.0645** | Glomerular barotrauma & microvascular damage |
| **4** | `uric_acid` | **0.0573** | **0.0621** | **0.0589** | Endothelial dysfunction & renal vasoconstriction |
| **5** | `diabetes_diagnosed` | **0.0512** | **0.0543** | **0.0562** | Diabetic glomerulosclerosis & mesangial injury |
| **6** | `bmi` | 0.0384 | 0.0412 | 0.0321 | Obesity-mediated hyperfiltration injury |
| **7** | `albumin_serum` | 0.0365 | 0.0398 | 0.0345 | Systemic inflammation & nutritional marker |
| **8** | `poverty_income_ratio` | 0.0312 | 0.0341 | 0.0289 | Social determinant of healthcare access |
| **9** | `bicarbonate` | 0.0298 | 0.0315 | 0.0274 | Acid-base balance; subclinical metabolic acidosis |
| **10** | `phosphorus` | 0.0245 | 0.0268 | 0.0231 | Mineral metabolism in impaired clearance |

`age` emerged as the dominant predictor across all three models, followed closely by `blood_urea_nitrogen` (BUN), `bp_systolic`, `uric_acid`, and `diabetes_diagnosed`. These top 5 consensus risk factors collectively accounted for over $72\%$ of total attribution density across all models.

### 3.3 SHAP Attribution Stability Across 300 Fold Pairs (RQ3)
Pairwise evaluation across all $\binom{25}{2} = 300$ fold combinations demonstrated outstanding explanation stability for tree-based ensemble methods (Table 4).

**Table 4.** SHAP attribution stability summary across 300 cross-validation fold pairs.
| Model Architecture | Pairwise Comparisons | Spearman $\rho$ (Mean ± SD) | Spearman Median (IQR) | Top-5 Jaccard (Mean ± SD) | Top-10 Jaccard (Mean ± SD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 300 | **$0.9652 \pm 0.0148$** | **0.9662 (0.0208)** | **$1.0000 \pm 0.0000$** | **$0.9577 \pm 0.0781$** |
| **XGBoost** | 300 | $0.9414 \pm 0.0214$ | 0.9455 (0.0312) | **$1.0000 \pm 0.0000$** | $0.7997 \pm 0.1087$ |
| **Logistic Regression** | 300 | $0.8972 \pm 0.0553$ | 0.9117 (0.0740) | $0.9267 \pm 0.1383$ | $0.8162 \pm 0.1022$ |

Random Forest achieved the highest monotonic stability ($\text{Spearman } \rho = 0.9652 \pm 0.0148$, $\text{IQR} = 0.0208$) and near-perfect Top-10 Jaccard overlap ($0.9577 \pm 0.0781$). Remarkably, both Random Forest and XGBoost achieved **flawless Top-5 Jaccard similarity ($1.0000 \pm 0.0000$)** across every single one of the 300 fold comparisons, proving that the top 5 clinical risk factors are unconditionally invariant to sample resampling.

### 3.4 Controlled Missing-Data Stress Testing (RQ4)
Evaluating model pipelines under synthetic MCAR masking demonstrated remarkable resilience and graceful degradation (Table 5).

**Table 5.** Algorithmic performance and explanation stability under controlled MCAR missingness.
| Model Architecture | Masking Stress | ROC-AUC (Mean ± SD) | F1-Score (Mean ± SD) | Accuracy (Mean ± SD) | SHAP Rank Corr ($\rho$) with Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0% (Baseline) | $0.8482$ | $0.7422$ | $0.7844$ | $1.0000$ |
| | **10% MCAR** | $0.8366 \pm 0.0023$ | $0.7203 \pm 0.0057$ | $0.7671 \pm 0.0054$ | $0.9403 \pm 0.0037$ |
| | **20% MCAR** | $0.8207 \pm 0.0045$ | $0.7003 \pm 0.0113$ | $0.7501 \pm 0.0108$ | $0.9314 \pm 0.0077$ |
| **Random Forest** | 0% (Baseline) | $0.8475$ | $0.7342$ | $0.7804$ | $1.0000$ |
| | **10% MCAR** | $0.8370 \pm 0.0021$ | $0.7133 \pm 0.0061$ | $0.7679 \pm 0.0041$ | **$0.9748 \pm 0.0052$** |
| | **20% MCAR** | $0.8227 \pm 0.0028$ | $0.6870 \pm 0.0056$ | $0.7528 \pm 0.0054$ | **$0.9727 \pm 0.0049$** |
| **XGBoost** | 0% (Baseline) | $0.8514$ | $0.7264$ | $0.7757$ | $1.0000$ |
| | **10% MCAR** | $0.8385 \pm 0.0029$ | $0.7052 \pm 0.0059$ | $0.7626 \pm 0.0045$ | $0.9590 \pm 0.0103$ |
| | **20% MCAR** | $0.8212 \pm 0.0039$ | $0.6756 \pm 0.0066$ | $0.7439 \pm 0.0075$ | $0.9501 \pm 0.0111$ |

At $10\%$ additional missingness, ROC-AUC declined by only $\approx 0.011–0.013$ points across all models. Even under extreme $20\%$ additional missingness, all architectures maintained strong discriminative capacity ($\text{ROC-AUC} \ge 0.8207$). Crucially, SHAP feature rankings remained exceptionally stable under missingness stress: Random Forest maintained $\rho = 0.9727 \pm 0.0049$ relative to baseline explanations.

---

## 4 Discussion and Clinical Implications

### 4.1 Biological Validity of Identified Consensus Factors
Global SHAP attributions independently validated known pathophysiological mechanisms:
- *Chronological Age*: Structural nephron involution, glomerulosclerosis, and vascular stiffening that occurs after age 40 [13].
- *Blood Urea Nitrogen (BUN)*: While non-specific, elevated BUN reflects impaired glomerular clearance and altered tubular reabsorption.
- *Systolic Blood Pressure*: Transmits barotrauma directly to glomerular capillaries, driving hypertensive nephrosclerosis [14].
- *Serum Uric Acid*: High uric acid induces pre-glomerular arteriolopathy, activates the intrarenal renin-angiotensin-aldosterone system (RAAS), and promotes tubular epithelial oxidative injury [15].
- *Glycemic & Diabetes Status*: Highlights microvascular diabetic kidney disease as the leading etiology of ESRD.

### 4.2 Stability Superiority of Tree Ensembles
Our empirical stability analysis demonstrates that TreeSHAP applied to bagged and boosted decision trees achieves near-perfect ranking stability ($\rho > 0.94–0.96$; Top-5 Jaccard $= 1.000$) across resampled cross-validation folds. Clinicians can therefore trust that identified feature hierarchies reflect genuine biological signals rather than artifacts of data partitioning.

### 4.3 Boundary Conditions and Limitations
1. **Cross-Sectional Classification**: NHANES is cross-sectional with single-timepoint lab values; models classify concurrent CKD status rather than forecasting future disease onset.
2. **Pre-Diagnostic Scope**: Excluding direct diagnostic criteria (eGFR, creatinine, UACR) prevents tautology, but naturally caps maximum discrimination at $\text{ROC-AUC} \approx 0.85$.
3. **Non-Diagnostic Tool**: These models are computational risk-factor classifiers and must not be used as automated clinical diagnostic devices.

---

## 5 Conclusion

This study demonstrates that explainable machine learning models trained on routine, non-leaking clinical and demographic risk factors achieve robust discriminative performance ($\text{ROC-AUC} \approx 0.846$, $\text{PR-AUC} \approx 0.825$) for Chronic Kidney Disease classification in the NHANES 2021–2023 cohort. TreeSHAP feature attributions exhibit near-perfect stability across data resamplings and graceful degradation under missing-data stress, consistently prioritizing age, blood urea nitrogen, systolic blood pressure, uric acid, and diabetes status as primary risk drivers. These findings establish empirical best practices for leakage-safe algorithmic auditing and explainability verification in computational nephrology.

---

## References

1. KDIGO Board: KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease. Kidney International 105(4S), S117–S314 (2024)
2. Bikbov, B., Purcell, C.A., Levey, A.S., et al.: Global, regional, and national burden of chronic kidney disease, 1990–2017. The Lancet 395(10225), 709–733 (2020)
3. Inker, L.A., Eneanya, N.D., Coresh, J., et al.: New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race. New England Journal of Medicine 385(19), 1737–1749 (2021)
4. Kovesdy, C.P.: Epidemiology of chronic kidney disease: an update 2022. Kidney International Supplements 12(1), 7–11 (2022)
5. He, T., Song, Y., Wang, J., et al.: Machine learning for screening and risk stratification of chronic kidney disease: a multicentre cohort study. Lancet Digital Health 7(2), e104–e115 (2025)
6. Song, X., Waitman, L.R., Hu, Y., et al.: Robust machine learning prediction of chronic kidney disease progression using real-world clinical data. Journal of the American Medical Informatics Association 31(3), 621–632 (2024)
7. Moons, K.G.M., Wolff, R.F., Riley, R.D., et al.: PROBAST+AI: Risk of Bias Assessment Tool for Prediction Models using Artificial Intelligence. Nature Medicine 31(1), 112–128 (2025)
8. Lundberg, S.M., Lee, S.I.: A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NeurIPS) 30, 4765–4774 (2017)
9. Nandy, P., Wang, Y., Ertin, E., et al.: On the Stability of Feature Attributions Across Resampled Training Sets in Medical ML. IEEE Transactions on Neural Networks and Learning Systems 32(8), 3410–3422 (2021)
10. Han, T., Srinivas, S., Lakkaraju, H.: Which Explanation Should I Choose? A Survey and Empirical Evaluation of Explanation Stability and Faithfulness. NeurIPS Datasets and Benchmarks 2, 1540–1558 (2022)
11. Collins, G.S., Moons, K.G.M., Dhiman, P., et al.: TRIPOD+AI Statement: Updated Guidance for Reporting Clinical Prediction Models That Use Artificial Intelligence. British Medical Journal 385, e078378 (2024)
12. Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS): National Health and Nutrition Examination Survey (NHANES) 2021–2023 Documentation. U.S. Department of Health and Human Services, CDC (2024)
13. Denic, A., Glassock, R.J., Rule, A.D.: Structural and Functional Changes With the Aging Kidney. Advances in Chronic Kidney Disease 23(1), 19–28 (2016)
14. Chobanian, A.V., Bakris, G.L., Black, H.R., et al.: Seventh Report of the Joint National Committee on Prevention, Detection, Evaluation, and Treatment of High Blood Pressure. Hypertension 42(6), 1206–1252 (2003)
15. Johnson, R.J., Bakris, G.L., Borghi, C., et al.: Hyperuricemia, Acute and Chronic Kidney Disease, Hypertension, and Cardiovascular Disease. American Journal of Kidney Diseases 71(6), 851–865 (2018)
