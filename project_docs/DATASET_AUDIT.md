# DATASET AUDIT: NHANES 2021–2023 CKD Classification Dataset

## 1. Dataset Overview & Provenance
- **Dataset File**: `CKD_NHANES_2021_2023.csv`
- **Source**: Kaggle dataset "CKD-NHANES 2021–2023: Staged Kidney Disease" by Ali Taqi Shah, derived from the CDC/NCHS National Health and Nutrition Examination Survey (NHANES) August 2021–August 2023 continuous cycle.
- **CDC Documentation Reference**: CDC NHANES 2021–2023 Data & Laboratory Documentation (`wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023`).
- **Total Raw Records**: 11,933 participants
- **Total Columns**: 29 variables
- **Duplicate Records**: 0 exact duplicate rows; 11,933 unique `participant_id` values.

---

## 2. Target Variable Audit & Critical Finding
### 2.1 Raw Target Distributions
In the raw CSV file:
- `ckd_stage` distribution:
  - `Unknown`: 5,607 (47.0%)
  - `No CKD`: 3,592 (30.1%)
  - `Stage 2 (Mildly Decreased)`: 1,875 (15.7%)
  - `Stage 1 (Kidney Damage)`: 382 (3.2%)
  - `Stage 3a (Mild-Moderate)`: 341 (2.9%)
  - `Stage 3b (Moderate-Severe)`: 97 (0.8%)
  - `Stage 4 (Severely Decreased)`: 28 (0.2%)
  - `Stage 5 (Kidney Failure)`: 11 (0.1%)
- `ckd_present` distribution:
  - Class `1`: 8,341 (69.9%)
  - Class `0`: 3,592 (30.1%)

### 2.2 Target Derivation & Cohort Selection
- **Audit Discovery**: The dataset creator constructed `ckd_present` via `(ckd_stage != 'No CKD')`. Consequently, 5,607 participants who did not have kidney lab measurements (pediatric participants $<12$ years or individuals without biochemistry profiles) and thus had `ckd_stage == 'Unknown'` were artificially assigned `ckd_present = 1`.
- **Methodological Correction**:
  - Participants with `ckd_stage == 'Unknown'` have unmeasured/unverifiable CKD status and must be excluded from the evaluable cohort.
  - The valid research cohort consists of **$N = 6,326$** participants with verified laboratory measurements and complete staging.
- **Valid Cohort Target Breakdown ($N=6,326$)**:
  - Class `0` (No CKD): 3,592 (56.78%)
  - Class `1` (Documented CKD Stages 1–5): 2,734 (43.22%)
  - Class Balance: 1.31:1 (well-balanced, no extreme rarity).

---

## 3. Comprehensive Feature Audit Table

| Feature Name | Raw Data Type | Missingness (Raw, N=11,933) | Missingness (Valid, N=6,326) | Clinical / Survey Meaning | Target Leakage Risk | Decision | Rationale |
|---|---|---|---|---|---|---|---|
| `participant_id` | Float | 0 (0.0%) | 0 (0.0%) | NHANES SEQN participant identifier | None (ID) | **Exclude** | Non-predictive administrative identifier. |
| `age` | Float | 0 (0.0%) | 0 (0.0%) | Participant age in years (12–80) | None | **Include** | Key demographic risk factor for declining GFR. |
| `gender` | Object | 0 (0.0%) | 0 (0.0%) | Biological sex (Male, Female) | None | **Include** | Demographic predictor and determinant of baseline muscle mass/creatinine. |
| `ethnicity` | Object | 0 (0.0%) | 0 (0.0%) | Race/Hispanic origin category (6 levels) | None | **Include** | Epidemiological risk factor and socioeconomic covariate. |
| `education_level` | Float | 4,139 (34.7%) | 883 (14.0%) | Educational attainment (1–5, 9=unknown) | None | **Include** | Socioeconomic status indicator. |
| `poverty_income_ratio`| Float | 2,041 (17.1%) | 793 (12.5%) | Ratio of family income to poverty threshold | None | **Include** | Continuous socioeconomic status metric. |
| `bmi` | Float | 3,462 (29.0%) | 81 (1.3%) | Body Mass Index ($\text{kg/m}^2$) | None | **Include** | Obesity marker, established metabolic risk factor. |
| `weight_kg` | Float | 3,179 (26.6%) | 67 (1.1%) | Measured body weight (kg) | None | **Include** | Anthropometric measure. |
| `height_cm` | Float | 3,434 (28.8%) | 61 (1.0%) | Measured standing height (cm) | None | **Include** | Anthropometric measure. |
| `bp_systolic` | Float | 4,416 (37.0%) | 193 (3.1%) | Systolic blood pressure (mmHg) | None | **Include** | Major vascular risk factor for hypertensive nephropathy. |
| `bp_diastolic` | Float | 4,416 (37.0%) | 193 (3.1%) | Diastolic blood pressure (mmHg) | None | **Include** | Vascular and hemodynamic risk factor. |
| `serum_creatinine` | Float | 5,607 (47.0%) | 0 (0.0%) | Serum creatinine ($\text{mg/dL}$) | **High (Direct)** | **Exclude** | Direct mathematical input to 2021 CKD-EPI eGFR used to define the target label. |
| `blood_urea_nitrogen`| Float | 5,607 (47.0%) | 3 (0.05%) | Blood urea nitrogen (BUN, $\text{mg/dL}$) | Low | **Include** | General renal/nitrogenous waste marker, not part of KDIGO stage definition. |
| `albumin_serum` | Float | 5,567 (46.7%) | 0 (0.0%) | Serum albumin ($\text{g/dL}$) | Low | **Include** | General nutritional and liver/protein metabolism marker. |
| `phosphorus` | Float | 5,611 (47.0%) | 19 (0.3%) | Serum phosphorus ($\text{mg/dL}$) | Low | **Include** | Mineral metabolism marker affected by advanced renal clearance. |
| `bicarbonate` | Float | 5,608 (47.0%) | 3 (0.05%) | Serum bicarbonate / $\text{CO}_2$ ($\text{mmol/L}$) | Low | **Include** | Acid-base balance marker. |
| `calcium` | Float | 5,571 (46.7%) | 4 (0.06%) | Total serum calcium ($\text{mg/dL}$) | Low | **Include** | Mineral metabolism covariate. |
| `uric_acid` | Float | 5,604 (47.0%) | 0 (0.0%) | Serum uric acid ($\text{mg/dL}$) | Low | **Include** | Purine metabolite associated with metabolic and vascular disease. |
| `urine_creatinine` | Float | 3,779 (31.7%) | 123 (1.9%) | Urine creatinine ($\text{mg/dL}$) | **High (Direct)** | **Exclude** | Direct denominator of UACR used to define KDIGO Stage 1 CKD. |
| `urine_albumin` | Float | 3,780 (31.7%) | 124 (2.0%) | Urine albumin ($\text{ug/mL}$) | **High (Direct)** | **Exclude** | Direct numerator of UACR used to define KDIGO Stage 1 CKD. |
| `albumin_creatinine_ratio`| Float | 3,780 (31.7%) | 124 (2.0%) | Urinary Albumin-to-Creatinine Ratio (UACR) | **High (Direct)** | **Exclude** | Direct diagnostic threshold ($\ge 30\text{ mg/g}$) used to define Stage 1 CKD. |
| `diabetes_diagnosed`| Float | 193 (1.6%) | 0 (0.0%) | Self-reported doctor diagnosis of diabetes | None | **Include** | Leading etiology of chronic kidney disease (diabetic nephropathy). |
| `insulin_use` | Float | 10,852 (90.9%) | 5,560 (87.9%)| Use of insulin for diabetes | None | **Include** | Indicator of insulin-dependent / severe diabetes. |
| `diabetes_pills` | Float | 9,652 (80.9%) | 4,682 (74.0%)| Use of oral hypoglycemics | None | **Include** | Indicator of pharmacologically managed type 2 diabetes. |
| `ever_smoked` | Float | 3,798 (31.8%) | 670 (10.6%) | Smoked $\ge 100$ cigarettes in lifetime | None | **Include** | Cardiovascular and endothelial risk factor. |
| `current_smoker` | Float | 8,690 (72.8%) | 4,089 (64.6%)| Current smoking frequency | None | **Include** | Active lifestyle risk factor. |
| `egfr` | Float | 5,607 (47.0%) | 0 (0.0%) | Estimated GFR ($\text{mL/min/1.73 m}^2$) | **High (Deterministic)**| **Exclude** | Ground-truth mathematical formula used to establish CKD stages 2–5. |
| `ckd_stage` | Object | 0 (0.0%) | 0 (0.0%) | KDIGO Staging label | **Direct Target** | **Exclude** | Staging target variable. |
| `ckd_present` | Int | 0 (0.0%) | 0 (0.0%) | Binary CKD indicator ($0=\text{No}, 1=\text{Yes}$) | **Target** | **Target** | Primary binary classification target. |

---

## 4. Summary of Predictor Partition
- **Total Predictors Included**: 21 features
  - Continuous/Numeric (14): `age`, `poverty_income_ratio`, `bmi`, `weight_kg`, `height_cm`, `bp_systolic`, `bp_diastolic`, `blood_urea_nitrogen`, `albumin_serum`, `phosphorus`, `bicarbonate`, `calcium`, `uric_acid`, `education_level` (treated as numeric/ordinal with median imputation or categorical; here configured with explicit imputation).
  - Categorical (7): `gender`, `ethnicity`, `diabetes_diagnosed`, `insulin_use`, `diabetes_pills`, `ever_smoked`, `current_smoker`.
- **Total Features Excluded Due to Leakage / Identifier**: 7 features (`participant_id`, `serum_creatinine`, `urine_creatinine`, `urine_albumin`, `albumin_creatinine_ratio`, `egfr`, `ckd_stage`).
