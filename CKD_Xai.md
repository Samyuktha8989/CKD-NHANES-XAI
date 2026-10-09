COMPLETE STUDY PLAN
Explainable Machine Learning for Chronic Kidney Disease Classification
Predictive performance, SHAP explanation stability, and missing-data sensitivity

A reproducible, small-dataset experimental study plan
Prepared 9 October 2026
Executive decision
Study type: original experimental machine-learning study using a public, retrospective tabular dataset. The proposed differentiator is not simply training classifiers and plotting SHAP. It is to quantify how consistent SHAP feature-attribution rankings remain across resampled training folds and how explanations and performance change under a predefined missingness stress test.
Proposed title: “Explainable Machine Learning for Chronic Kidney Disease Classification: A Study of SHAP Attribution Stability and Missing-Data Sensitivity.”
Novelty caution: this is a candidate contribution, not a guarantee that the exact combination has never been published. A targeted novelty audit is mandatory before claiming novelty. The UCI CKD dataset is small and widely reused, so the manuscript must describe the work as a methodological robustness study and must not claim clinical readiness or prediction of future CKD onset.
Critical correction to the earlier draft
Do not use the mock-data section from the pasted script. It generates random synthetic records and random labels. Those records are not the UCI dataset and cannot support scientific results. Download the real UCI data and verify its labels, columns, missing values, and sample count before training any model. The pasted script also uses only a subset of predictors; the final analysis should document any excluded variables and use the dataset’s real target column.
1. Research question and objectives
Primary question: How stable are SHAP-derived feature-attribution rankings for CKD classification across resampled training data, and how does controlled additional missingness affect predictive performance and explanation stability?
Specific research questions
ID
Question
Analysis
RQ1
How does predictive performance vary among Logistic Regression, Random Forest, and XGBoost?
Repeated stratified cross-validation; discrimination, classification, and calibration metrics.
RQ2
How consistent are SHAP feature rankings across resampled training folds?
Spearman rank correlation; top-5/top-10 Jaccard overlap.
RQ3
Do different model families identify similar influential original clinical features?
Cross-model ranking agreement after grouping one-hot encoded features.
RQ4
How does controlled additional missingness affect performance and SHAP explanations?
10% and 20% masking scenarios on held-out fold inputs, repeated with fixed seeds.

Objectives
    • Audit and document the public dataset, its target definition, feature types, and missingness.
    • Compare three prespecified classification models using a common, leakage-resistant pipeline.
    • Report performance across repeated stratified cross-validation rather than relying on one favorable split.
    • Use SHAP for global and local explanations of held-out predictions.
    • Quantify explanation stability across folds and model families.
    • Run a controlled missingness sensitivity experiment and document limitations.
2. Dataset choice and acceptance criteria
Primary dataset: UCI Machine Learning Repository, Chronic Kidney Disease (dataset ID 336; https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease). The repository describes 400 instances, 24 predictors, a classification task, and missing values. The label represents recorded CKD/non-CKD status; it is not longitudinal evidence of future disease onset.
Why this dataset?
    • It is public, small enough for a reproducible Colab notebook, and contains mixed numeric/categorical clinical variables.
    • It supports a manageable binary classification exercise without recruiting participants or collecting new data.
    • Its small size and repeated use in prior publications are limitations; these must be stated, and novelty must come from a carefully defined analysis rather than the dataset itself.
Dataset audit before modelling
    • Record the UCI page, access date, citation, dataset files/version, and any licence/usage terms.
    • Confirm the exact feature matrix and target column from the official repository; do not assume downloaded column names.
    • Normalize whitespace, tabs, and missing-value markers such as '?', empty strings, or blanks.
    • Verify target labels and class counts; document records removed only when the target is invalid or missing.
    • Report feature type, number and percentage missing per variable, duplicates, and basic descriptive statistics.
    • Investigate duplicates before deciding whether to remove them; never delete rows automatically without documenting the rationale.
    • Check for target leakage, identifiers, and variables that may directly encode the label.
    • Do not fabricate, augment, or substitute random mock data for the real dataset.
3. Study design and validation
Primary design: repeated stratified 5-fold cross-validation with 5 repeats (25 held-out fold evaluations per model). Use fixed random seeds and identical folds across models. Predefine model settings rather than running a large hyperparameter search on this small dataset. The repeated folds overlap and are not 25 independent cohorts; report means and standard deviations descriptively and avoid naive significance tests that treat all folds as independent.
For each fold, fit every imputer, encoder, scaler, and classifier using that fold’s training data only. Generate predictions and SHAP explanations for that fold’s held-out records. This makes the cross-validation analysis the primary estimate and provides held-out explanations for the stability analysis. A separate single 80/20 split is not required for the primary design; if one is added, it must be clearly designated as a secondary holdout experiment and not used to tune models.
Preprocessing pipeline
Feature type
Preprocessing
Leakage safeguard
Numeric
Median imputation; StandardScaler for Logistic Regression. Tree models do not require scaling, but a shared preprocessing design may retain it if applied consistently.
Fit imputer/scaler within each training fold.
Categorical
Most-frequent imputation; OneHotEncoder(handle_unknown='ignore').
Fit imputer/encoder within each training fold.
Target
Map the verified CKD class to 1 and non-CKD to 0.
Never derive the target from predictors.
Feature selection
No global outcome-driven selection in the primary analysis.
Any later selection must occur inside training folds.

Use median/most-frequent imputation as a transparent baseline. KNN or iterative imputation can be explored only as a prespecified sensitivity analysis if time permits; do not switch methods after seeing which produces the most attractive result.
4. Models and configurations
Model
Purpose
Starting configuration
Logistic Regression
Linear baseline; useful contrast to tree models.
L2 regularization; max_iter=2000; standardize numeric variables; class weighting only if justified and documented.
Random Forest
Nonlinear bagged-tree model; use TreeSHAP.
n_estimators=300; min_samples_leaf=2; random_state fixed; record all settings.
XGBoost
Gradient-boosted tree comparator; use TreeSHAP.
Conservative depth (e.g., 3); modest learning rate (e.g., 0.05); fixed number of estimators; record package version and settings.

Do not use the test-fold labels for model selection. Because the dataset is small, start with prespecified settings and avoid an extensive tuning search. If tuning is essential, it must be nested inside the training data, which increases complexity and runtime.
5. Performance metrics
Metric
Definition / purpose
Reporting
ROC-AUC
Discrimination across probability thresholds.
Mean ± SD across folds; primary metric.
Sensitivity / Recall
TP / (TP + FN); proportion of CKD cases detected.
Mean ± SD; clinically important.
Specificity
TN / (TN + FP); proportion of non-CKD cases correctly identified.
Mean ± SD.
Precision / PPV
TP / (TP + FP); fraction of positive predictions that are CKD.
Mean ± SD.
F1-score
Harmonic mean of precision and recall.
Mean ± SD.
Accuracy
Overall fraction of correct classifications.
Secondary; never report alone.
MCC
Matthews correlation coefficient; balanced summary of binary classification.
Recommended secondary metric.
Brier score
Mean squared error of predicted probabilities.
Recommended calibration-related metric; lower is better.
Confusion matrix
Counts of TP, FP, TN, FN.
Summarize fold-level counts carefully or show an explicitly defined aggregate.

If a decision threshold other than 0.5 is used, define how it is chosen using training data only. Do not choose the threshold to maximize test-set performance. Report calibration cautiously: a small dataset cannot support strong claims about probability calibration.
6. Explainable AI plan: SHAP
A. Global explanation
    • Compute SHAP values for held-out observations for every fold-trained model.
    • Report mean absolute SHAP values and a summary/beeswarm plot for each model or for the primary model, with the selection rule declared in advance.
    • Group one-hot encoded levels back to their original clinical variable before comparing feature rankings; otherwise categorical variables may be fragmented across dummy columns.
    • Show both feature importance magnitude and direction where the plot supports it. Avoid translating attribution into causal or independent clinical risk claims.
B. Local explanation
    • Generate SHAP waterfall plots for a small number of held-out examples, selected by a reproducible rule.
    • Include at least one true positive and one true negative if such examples occur; optionally include a false negative and false positive to audit errors.
    • Do not hand-pick only impressive examples. Record the fold and anonymized row index, and state that the plots explain model output rather than provide medical advice.
C. Explanation stability (primary contribution candidate)
Measure
How to calculate
Interpretation
Spearman rank correlation
Correlation between two folds’ ranks of mean absolute SHAP importance over the same original features.
Higher means more consistent rank ordering.
Top-k Jaccard similarity
Intersection/union of top-5 or top-10 feature sets across folds.
Higher means more overlap among highly ranked features.
Attribution magnitude variation
For each feature, summarize variation in mean absolute SHAP values across folds; normalize only with a documented rule if comparing scales.
Shows whether attribution magnitude varies across fitted models.
Cross-model agreement
Compare original-feature rankings across Logistic Regression, Random Forest, and XGBoost.
Identifies agreement or disagreement across model families; not proof of biological truth.

Compute pairwise fold comparisons and report the distribution (median, interquartile range, and/or mean ± SD). Because pairwise comparisons are dependent, do not treat them as independent samples for conventional significance tests. SHAP values from different model families may have different scales and semantics; rank agreement is more defensible than directly comparing raw magnitudes.
D. Missingness sensitivity experiment
    • Baseline: evaluate each fold model on its unmodified held-out fold, preserving naturally missing values handled by the fitted pipeline.
    • Stress tests: on the held-out inputs only, randomly mask 10% and 20% of currently observed predictor cells; repeat each scenario with multiple prespecified random seeds.
    • Use the already fitted training-fold imputer to transform masked data. Do not refit the model or imputer on masked test data.
    • Measure changes in ROC-AUC, sensitivity, specificity, F1, and SHAP ranking stability relative to the unmodified fold.
    • Describe the masking as a synthetic stress test, not a claim that the imposed missingness percentages represent real hospital missingness patterns.
7. Expected outputs (not expected findings)
Output
Contents
Manuscript location
Table 1
Dataset dimensions, target distribution, feature types, missingness.
Methods / Results
Table 2
Mean ± SD for each model and metric across repeated folds.
Results
Figure 1
ROC curves or fold-level performance distribution, with plotting protocol stated.
Results
Figure 2
Confusion-matrix summary with explicit aggregation method.
Results
Figure 3
SHAP global summary plot(s) for held-out fold predictions.
Results
Figure 4
Heatmap/distribution of Spearman and top-k Jaccard explanation stability.
Results
Table 3
Top-k feature overlap and ranking stability by model.
Results
Figure/Table 5
Metric and explanation changes under 10%/20% missingness stress tests.
Results
Figure 6
Reproducibly selected local SHAP waterfall examples.
Results / Discussion
Supplement
Code, package versions, seeds, preprocessing details, fold-level results, and data-source citation.
Reproducibility

These are deliverables to generate, not results that can be written in advance. Do not insert assumed accuracy values or assumed top features.
8. Manuscript sections and what to write
Section
Content to include
Title
Explainable ML for CKD classification; explicitly mention attribution stability and missing-data sensitivity if these analyses are completed.
Abstract
Background, objective, dataset, models, validation, SHAP stability analysis, actual key results, limitations. Write the final version last.
1. Introduction
CKD context; ML for classification; explainability motivation; why stability and missingness matter; precise research questions and contributions.
2. Related Work
CKD ML studies; explainable CKD studies; SHAP; explanation robustness/stability; clinical prediction reporting and validation.
3. Materials and Methods
Dataset source and labels; inclusion/exclusion; preprocessing; models; CV design; metrics; SHAP extraction; ranking stability; missingness experiment.
4. Experimental Setup
Environment, package versions, random seeds, model parameters, computational details, reproducibility.
5. Results
Dataset audit, cross-validated metrics, SHAP results, explanation stability, missingness sensitivity. No interpretation beyond observed results.
6. Discussion
Answer each RQ; compare with previous studies; discuss model/explanation disagreement; clinical plausibility without causal claims.
7. Limitations
Small and frequently reused dataset; no external validation; possible label/selection bias; synthetic masking; uncertainty and limited clinical generalizability.
8. Conclusion
Summarize only demonstrated findings and propose external validation on a representative independent cohort.
References
Approximately 20 carefully verified references grouped by disease, dataset, ML methods, XAI, and evaluation/reporting.

9. Novelty audit before claiming originality
No search can guarantee that a study idea has never appeared anywhere. Before stating a novelty claim, search Google Scholar, PubMed, Scopus (if available), Crossref, and the reference lists of the closest papers. Record the date, exact queries, and the closest matching studies.
Recommended exact search queries:
    • "chronic kidney disease" SHAP explanation stability
    • "CKD prediction" "feature attribution stability"
    • "chronic kidney disease" SHAP robustness missing data
    • "UCI CKD" repeated cross-validation explainability
    • "feature ranking stability" clinical machine learning SHAP
    • "explanation stability" tabular healthcare machine learning
If a paper already performs the same combination of resampled SHAP-ranking stability and missingness sensitivity on CKD data, do not claim the combination is new. Reframe as a replication/robustness study or choose a more distinctive question or dataset.
10. End-to-end implementation plan
Step
Action
Files/results to save
Completion check
1
Fetch the actual UCI dataset and inspect metadata.
raw data reference; dataset_audit.csv
400 records / expected columns verified against source; no mock data.
2
Clean whitespace/missing markers; define target; audit duplicates and types.
cleaned_data.csv; audit report
Class counts and missingness documented.
3
Build numeric/categorical preprocessing pipelines.
preprocessing_config.json or notebook cell
All transformations fit within folds.
4
Define three models with prespecified settings.
model_config.json
Parameters and versions recorded.
5
Run repeated stratified 5-fold CV (5 repeats).
fold_metrics.csv
All models use identical splits.
6
Compute SHAP for each fold-trained model on held-out rows.
fold_shap_values; fold_feature_importance.csv
No SHAP ranking uses training predictions only.
7
Aggregate one-hot features to original variables and calculate stability.
stability_pairwise.csv; stability_summary.csv
Same feature universe and aggregation rule across comparisons.
8
Run 10%/20% additional missingness stress tests.
missingness_results.csv
Masking seeds and evaluation protocol recorded.
9
Generate figures and tables.
PNG/PDF/SVG and CSV outputs
Captions, units, class positive definition, and aggregation rules documented.
10
Write the paper and verify citations.
manuscript.docx or LaTeX
All numerical claims trace to outputs; all references verified.
11
Run reporting and novelty audits.
TRIPOD+AI checklist; novelty search log
Limitations disclosed; no unsupported novelty or clinical claims.

11. Code architecture
Use Google Colab or a local Python environment. Suggested notebook cells:
    1. Imports, environment versions, random seeds.
    2. Fetch the official UCI dataset (e.g., ucimlrepo dataset ID 336); inspect the actual returned schema.
    3. Clean labels and missing markers; create a data audit table.
    4. Identify numeric and categorical columns from the verified schema.
    5. Create scikit-learn ColumnTransformer pipelines: median imputation + scaling for numeric inputs; most-frequent imputation + one-hot encoding for categorical inputs.
    6. Define Logistic Regression, Random Forest, and XGBoost pipelines.
    7. Run RepeatedStratifiedKFold with 5 splits and 5 repeats; save fold metrics.
    8. For each fitted fold model, transform held-out rows using that fold’s fitted preprocessor, compute SHAP values, and map transformed columns back to original variables.
    9. Calculate pairwise Spearman correlations, top-5/top-10 Jaccard similarities, and fold-level SHAP magnitude variation.
    10. Run held-out masking scenarios at 10% and 20%, with multiple seeds, without refitting the model or preprocessing.
    11. Generate plots, export CSVs, and zip the results folder.
Important implementation note: SHAP output formats differ by SHAP version and classifier. Test extraction on one fold first, confirm which output corresponds to class CKD, then implement the loop. Store the model version, SHAP version, class index, transformed feature names, and aggregation rule.
12. Time plan
Work block
Estimated effort
Deliverable
Dataset audit and clean pipeline
1–2 hours
Verified dataset and preprocessing
Models and repeated CV
1–3 hours
Fold-level metrics
SHAP extraction and stability
2–4 hours
Attributions and ranking stability
Missingness stress test and plots
1–3 hours
Robustness results
Novelty audit and reference verification
2–4 hours
Closest-paper comparison and checked bibliography
Manuscript drafting and compliance
4–8+ hours
Working manuscript with actual results

These are planning estimates, not guarantees. Package compatibility, dataset inconsistencies, and SHAP extraction can take longer. Do not write a final Results section until the experiment has run.
13. Publication and integrity notes
    • This is an original experimental study, not a review paper.
    • The official UCI dataset is publicly accessible; cite it and follow its terms.
    • Do not claim prospective early prediction, clinical readiness, causality, or external validity from this dataset.
    • Do not invent metrics, feature rankings, citations, sample sizes, or screening results.
    • Use the target conference/journal template only after the scientific manuscript is stable. Springer LNCS and IEEE formats are not interchangeable.
    • Use TRIPOD+AI as a reporting guide; it is not proof that a model is clinically valid or free of bias.
References (20 maximum; grouped by role)
A. Disease context and dataset
[1] KDIGO CKD Work Group, “KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease,” Kidney International, 2024. https://kdigo.org/guidelines/ckd-evaluation-and-management/
[2] A. S. Levey et al., “A new equation to estimate glomerular filtration rate,” Annals of Internal Medicine, vol. 150, no. 9, pp. 604–612, 2009. doi:10.7326/0003-4819-150-9-200905050-00006.
[3] L. Rubini, P. Soundarapandian, and P. Eswaran, “Chronic Kidney Disease,” UCI Machine Learning Repository, 2015. doi:10.24432/C5G020. https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
[4] L. Sanmarchi et al., “Predict, diagnose, and treat chronic kidney disease with machine learning: a systematic literature review,” Journal of Nephrology, 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10227138/
[5] E. Webster et al., “Chronic Kidney Disease,” The Lancet, vol. 389, pp. 1238–1252, 2017. doi:10.1016/S0140-6736(16)32064-5.
B. Machine-learning methods and evaluation
[6] L. Breiman, “Random Forests,” Machine Learning, vol. 45, pp. 5–32, 2001. doi:10.1023/A:1010933404324.
[7] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016. doi:10.1145/2939672.2939785.
[8] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011. https://jmlr.org/papers/v12/pedregosa11a.html
[9] R. Kohavi, “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection,” in Proc. 14th International Joint Conference on Artificial Intelligence, 1995, pp. 1137–1143.
[10] T. Saito and M. Rehmsmeier, “The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,” PLOS ONE, vol. 10, no. 3, e0118432, 2015. doi:10.1371/journal.pone.0118432.
[11] B. W. Matthews, “Comparison of the predicted and observed secondary structure of T4 phage lysozyme,” Biochimica et Biophysica Acta, vol. 405, no. 2, pp. 442–451, 1975. doi:10.1016/0005-2795(75)90109-9.
C. Explainable AI and explanation robustness
[12] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” in Advances in Neural Information Processing Systems, vol. 30, 2017. https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
[13] M. T. Ribeiro, S. Singh, and C. Guestrin, “‘Why Should I Trust You?’ Explaining the Predictions of Any Classifier,” in Proc. ACM SIGKDD, 2016. doi:10.1145/2939672.2939778.
[14] A. B. Arrieta et al., “Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AI,” Information Fusion, vol. 58, pp. 82–115, 2020. doi:10.1016/j.inffus.2019.12.012.
[15] A. Ghorbani, A. Abid, and J. Zou, “Interpretation of Neural Networks Is Fragile,” in Proc. AAAI Conference on Artificial Intelligence, vol. 33, no. 1, pp. 3681–3688, 2019. doi:10.1609/aaai.v33i01.33013681.
[16] E. M. Chouit, M. Rachdi, M. Bellafkih, and B. Raouyane, “Interpretable machine learning for chronic kidney disease prediction: Insights from SHAP and LIME analyses,” PLOS ONE, vol. 21, no. 2, e0343205, 2026. doi:10.1371/journal.pone.0343205.
[17] C. Molnar, Interpretable Machine Learning: A Guide for Making Black Box Models Explainable, online book. https://christophm.github.io/interpretable-ml-book/
D. Clinical prediction reporting and risk of bias
[18] G. S. Collins et al., “TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods,” BMJ, vol. 385, e078378, 2024. doi:10.1136/bmj-2023-078378.
[19] K. G. M. Moons et al., “PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods,” BMJ, 2025. https://www.bmj.com/content/388/bmj-2024-082505
[20] J. He, X. Wang, P. Zhu, X. Wang, Y. Zhang, J. Zhao, W. Sun, K. Hu, W. He, and J. Xie, “Identification and validation of an explainable early-stage chronic kidney disease prediction model: a multicenter retrospective study,” EClinicalMedicine, vol. 84, art. 103286, 2025. doi:10.1016/j.eclinm.2025.103286.
Reference verification note
Before submission, verify complete author lists, exact titles, volume/issue/pages, publication year, and DOI on publisher or repository pages. The list is capped at 20 numbered entries. The two recent CKD + XAI studies in [16] and [20] are especially important for the novelty audit: the proposed study must differ meaningfully from existing SHAP/LIME classification work and multicentre explainable CKD prediction.
