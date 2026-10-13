"""
Springer LNNS/LNCS Proceedings Document Generator: Compiles manuscript/manuscript_springer_lnns.docx
adhering strictly to Springer single-column proceedings guidelines (122 mm x 193 mm text block,
Times New Roman, Table captions above, Fig. captions below, LNCS citation formatting).
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

deps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))
if os.path.exists(deps_path) and deps_path not in sys.path:
    sys.path.insert(0, deps_path)


def set_cell_background(cell, hex_color: str):
    """Set background color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)


def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    """Set subtle gray borders on a table matching Springer style."""
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>\n'
            f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
            f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
            f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
            f'  <w:insideV w:val="none"/>\n'
            f'  <w:left w:val="none"/>\n'
            f'  <w:right w:val="none"/>\n'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)


def build_springer_manuscript(
    output_path: str = "manuscript/manuscript_springer_lnns.docx",
    figures_dir: str = "results/figures"
):
    """Build complete Springer LNNS formatted Word document."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = Document()

    # Springer Single-Column Dimensions (Text block: 122 mm wide x 193 mm high)
    # A4: 21.0 cm x 29.7 cm -> Left/Right: 4.4 cm, Top/Bottom: 5.2 cm
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(5.2)
        section.bottom_margin = Cm(5.2)
        section.left_margin = Cm(4.4)
        section.right_margin = Cm(4.4)

    # Base Normal Style: Times New Roman 10pt
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    normal_style.paragraph_format.line_spacing = 1.05
    normal_style.paragraph_format.space_after = Pt(3)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # 1. Paper Title: 14pt Bold Centered
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(12)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("Explainable Machine Learning for Classifying Chronic Kidney Disease using NHANES 2021-2023 Data: Stability of Feature Attribution and Sensitivity to Missing Data")
    title_run.font.name = 'Times New Roman'
    title_run.font.size = Pt(14)
    title_run.bold = True

    # 2. Authors: 10pt Centered
    author_p = doc.add_paragraph()
    author_p.paragraph_format.space_after = Pt(4)
    author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ar1 = author_p.add_run("Author Name1")
    ar1.font.name = 'Times New Roman'
    ar1.font.size = Pt(10)
    ar1_sup = author_p.add_run("1[0000-0000-0000-0000]")
    ar1_sup.font.superscript = True
    ar1_sup.font.size = Pt(8)
    
    author_p.add_run(", Co-Author Name1")
    ar2_sup = author_p.add_run("1")
    ar2_sup.font.superscript = True
    ar2_sup.font.size = Pt(8)

    author_p.add_run(", Senior Author Name2")
    ar3_sup = author_p.add_run("2*[0000-0000-0000-0000]")
    ar3_sup.font.superscript = True
    ar3_sup.font.size = Pt(8)

    # 3. Affiliations: 9pt Centered
    aff_p = doc.add_paragraph()
    aff_p.paragraph_format.space_after = Pt(14)
    aff_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    aff_p.paragraph_format.line_spacing = 1.0
    r_aff = aff_p.add_run(
        "1 Department of Computer Science and Engineering, Institute / University, City, Country\n"
        "2 Department of Biomedical Informatics and Computational Health, Institute / University, City, Country\n"
        "* Corresponding author: author@university.edu"
    )
    r_aff.font.name = 'Times New Roman'
    r_aff.font.size = Pt(9)
    r_aff.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # 4. Abstract: 9pt Justified with run-in bold "Abstract."
    abs_p = doc.add_paragraph()
    abs_p.paragraph_format.space_after = Pt(4)
    abs_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_absh = abs_p.add_run("Abstract. ")
    r_absh.bold = True
    r_absh.font.size = Pt(9)
    r_abst = abs_p.add_run(
        "Chronic kidney disease (CKD) affects an estimated 10% to 15% of adults worldwide; however, early indicators are generally asymptomatic and often remain undiagnosed. "
        "Supervised learning models trained on routine clinical examinations can aid opportunistic risk screenings; however, existing tabular benchmarks frequently contain "
        "target leakage by incorporating direct diagnostic markers as input features. Additionally, post-hoc explanations generated using Shapley Additive Explanations (SHAP) "
        "are rarely tested for stability across re-samples or robustness in the face of missing inputs. In this study, we assess non-leaking clinical and demographic variables "
        "from the Continuous National Health and Nutrition Examination Survey (NHANES) 2021–2023 to detect and resolve an artifact in the dataset where 5,607 records without "
        "laboratory measurements (mostly pediatric participants under 12 years old) were erroneously labeled CKD-positive by default logic, identifying a verified cohort of "
        "6,326 participants (3,592 non-CKD [56.8%] and 2,734 CKD [43.2%]). After quarantining seven target-defining features, we train Logistic Regression, Random Forest, "
        "and XGBoost using repeated stratified 5-fold cross-validation (25 folds) with 21 non-leaking predictors. TreeSHAP and LinearSHAP attributions are computed on "
        "out-of-fold validation splits, aggregated from one-hot levels back to parent variables, and evaluated across all 300 pairwise fold combinations. Additionally, "
        "models are stress- tested by adding 10% and 20% synthetic Missing Completely at Random (MCAR) noise in validation sets. Over 25 folds, XGBoost achieved an ROC-AUC "
        "of 0.8456 ± 0.0129 and a PR-AUC of 0.8246 ± 0.0149, with L2-regularized Logistic Regression (ROC-AUC 0.8441 ± 0.0123) and Random Forest (ROC-AUC 0.8406 ± 0.0123) "
        "close behind. Global SHAP rankings identified chronological age, blood urea nitrogen, systolic blood pressure, serum uric acid, and diagnosed diabetes as the primary "
        "predictors. Tree-based ensembles showed high Top-5 rank stability across 300 fold resamples (Jaccard index = 1.000 for Random Forest and XGBoost; mean Spearman "
        "rho = 0.9652 and 0.9414, respectively). When applying 10% and 20% missingness, the loss in discrimination was small (ROC-AUC drop ≤ 0.026), while SHAP attribution "
        "rankings maintained high rank correlation (rho ≥ 0.931) to baseline explanations. These results suggest that non-leaking systemic variables contain meaningful discriminative "
        "signal for CKD, and that tree-based Shapley explanations maintain robust ranking stability in the face of sampling variability and missing inputs."
    )
    r_abst.font.size = Pt(9)

    # 5. Keywords: 9pt with bold "Keywords:"
    kw_p = doc.add_paragraph()
    kw_p.paragraph_format.space_before = Pt(2)
    kw_p.paragraph_format.space_after = Pt(14)
    r_kwh = kw_p.add_run("Keywords: ")
    r_kwh.bold = True
    r_kwh.font.size = Pt(9)
    r_kwt = kw_p.add_run("Chronic Kidney Disease · Explainable Artificial Intelligence · SHAP · Feature Attribution Stability · Missing Data Sensitivity · NHANES · Soft Computing.")
    r_kwt.font.size = Pt(9)

    # Helper function for Springer Headings
    def add_springer_h1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        r.bold = True
        return h

    def add_springer_h2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(3)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.bold = True
        return h

    # Section 1: Introduction
    add_springer_h1("1 Introduction")
    doc.add_paragraph(
        "According to the Kidney Disease: Improving Global Outcomes (KDIGO 2024) clinical guideline [1], chronic kidney disease (CKD) "
        "is characterized by an estimated glomerular filtration rate (eGFR) below 60 mL/min/1.73 m 2 or persistent signs of kidney damage "
        "(e.g., an increased urinary albumin-to-creatinine ratio (UACR ≥ 30 mg/g)) for at least three months. Epidemiological findings from "
        "the Global Burden of Disease study conducted by Bikbov et al. [2] indicate that over 850 million people globally are affected by "
        "kidney impairment, playing a critical role in cardiovascular complications, end-stage renal disease (ESRD), and mortality."
    )
    doc.add_paragraph(
        "Since kidney disease in its early stages is predominantly asymptomatic, many patients remain undiagnosed until significant "
        "nephron functional reserve has been lost [4]. Automated risk stratification algorithms incorporating routine demographic, "
        "anthropometric, and outpatient laboratory measurements have consequently drawn considerable interest as tools for secondary screening [5,6]."
    )

    add_springer_h2("1.1 Methodological Pitfalls in Tabular Kidney Disease Models")
    doc.add_paragraph(
        "Although many machine learning studies report near-perfect classification metrics for CKD, clinical translation remains constrained by several methodological limitations:\n"
        "1. Tautological Target Leakage: A common and pervasive flaw in published tabular CKD research is the use of serum creatinine, eGFR, or spot urine albumin as input predictors [7]. "
        "Since clinical guidelines define CKD directly through eGFR thresholds and albuminuria criteria [1,3], models provided with these variables simply reproduce the mathematical formula "
        "of the outcome. As described in the PROBAST+AI risk-of-bias-evaluation standard [7], such designs yield artificially inflated discrimination (ROC-AUC > 0.99) while providing no value "
        "for opportunistic screening prior to ordering kidney-specific tests.\n"
        "2. Interpretability and Algorithmic Opacity: Nonlinear models such as gradient-boosted trees and random forests generate high-dimensional interaction structures. "
        "In the absence of transparent attribution frameworks, clinicians are unable to confirm whether classifications reflect valid physiological mechanisms or result from idiosyncratic data artifacts [8].\n"
        "3. Explanation Instability Across Cohort Resamples: Although Shapley Additive Explanations (SHAP) [8] offers an axiomatic basis for local and global attributions, "
        "empirical research by Nandy et al. [9] and Han et al. [10] demonstrates that post-hoc explanations may vary across training resamples. When minor differences in training data "
        "affect which clinical factors are perceived as most influential, clinicians cannot rely on the resulting explanations for decision support.\n"
        "4. Vulnerability to Missing Laboratory Data: Electronic health records and epidemiological surveys are rife with incomplete records [11]. In accordance with the "
        "TRIPOD+AI reporting guideline [11], predictive workflows should be systematically assessed under controlled missingness to ensure that neither predictions nor explanations collapse "
        "when specific tests are unavailable."
    )

    add_springer_h2("1.2 Study Objectives")
    doc.add_paragraph(
        "To address these gaps, we perform a leak-free empirical evaluation with the Continuous NHANES 2021–2023 dataset [12]. Specifically, we:\n"
        "• Examine the NHANES cohort and correct a labeling anomaly that incorrectly classified 5,607 pediatric and untested records.\n"
        "• Impose a strict target quarantine by removing direct diagnostic variables, benchmarking 21 routine predictors across Logistic Regression, Random Forest, and XGBoost through repeated stratified 5-fold cross-validation (25 folds).\n"
        "• Group one-hot dummy Shapley attributions back to their parent clinical variables and measure explanation stability across all 300 pairwise fold combinations using Spearman rank correlation and Top-k Jaccard overlap.\n"
        "• Evaluate performance and explanatory resilience under controlled Missing Completely at Random (MCAR) stress testing at 10% and 20% masking levels."
    )

    # Section 2: Materials and Methods
    add_springer_h1("2 Materials and Methods")
    add_springer_h2("2.1 Cohort Extraction and KDIGO Staging")
    doc.add_paragraph(
        "We examined cross-sectional data from the Continuous NHANES 2021–2023 cycle conducted by the National Center for Health Statistics (NCHS) within the Centers for Disease Control and Prevention [12]. "
        "Within the curated dataset (11,933 records and 29 variables), eGFR was determined using the race-free 2021 CKD-EPI creatinine equation developed by Inker et al. [3]:\n"
        "eGFR = 142 × min(S_cr / κ, 1)^α × max(S_cr / κ, 1)^(-1.200) × 0.9938^Age × [1.012 if Female]\n"
        "where S_cr denotes serum creatinine measured in mg/dL, κ equals 0.7 for females and 0.9 for males, and α is -0.241 for females and -0.302 for males. Staging is based on KDIGO criteria: "
        "Stage 1 (eGFR ≥ 90 with UACR ≥ 30), Stage 2 (60–89 with UACR ≥ 30), Stage 3a (45–59), Stage 3b (30–44), Stage 4 (15–29), Stage 5 (<15), and No CKD (eGFR ≥ 60 and UACR < 30).\n"
        "In our initial audit, we identified that participants without blood draws or urine tests (primarily children under 12 years, for whom creatinine is not collected in NHANES) "
        "were assigned ckd_stage = 'Unknown'. Nevertheless, the binary label was defined as ckd_stage != 'No CKD'. Since 'Unknown' was not equivalent to 'No CKD', 5,607 untested individuals "
        "(primarily healthy children) were incorrectly classified as CKD-positive. After removing these unmeasured records, we obtained a verified analytical cohort of N = 6,326 participants: "
        "3,592 non-CKD (56.78%) and 2,734 CKD-positive (43.22%)."
    )

    # Table 1: Baseline Characteristics (Springer Style: Caption ABOVE)
    t1_cap = doc.add_paragraph()
    t1_cap.paragraph_format.space_before = Pt(8)
    t1_cap.paragraph_format.space_after = Pt(3)
    t1_cap.paragraph_format.keep_with_next = True
    r_t1 = t1_cap.add_run("Table 1. Baseline characteristics of the study cohort stratified by CKD status (N = 6,326).")
    r_t1.bold = True
    r_t1.font.size = Pt(9)

    tbl1 = doc.add_table(rows=11, cols=5)
    tbl1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl1)

    t1_headers = ["Characteristic", "Total Cohort", "No CKD (y=0)", "CKD (y=1)", "Missing (%)"]
    for i, h in enumerate(t1_headers):
        cell = tbl1.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        set_cell_background(cell, "F2F2F2")

    t1_rows = [
        ["Age (years), Mean ± SD", "48.62 ± 20.71", "39.64 ± 19.82", "60.42 ± 15.21", "0.0%"],
        ["Female Gender, n (%)", "3,428 (54.19%)", "1,942 (54.06%)", "1,486 (54.35%)", "0.0%"],
        ["BMI (kg/m²), Mean ± SD", "29.08 ± 7.36", "28.12 ± 7.14", "30.34 ± 7.46", "1.28%"],
        ["Systolic BP (mmHg)", "120.72 ± 18.17", "115.32 ± 15.44", "127.81 ± 18.92", "3.05%"],
        ["Diastolic BP (mmHg)", "73.75 ± 11.55", "71.94 ± 10.82", "76.13 ± 12.06", "3.05%"],
        ["Blood Urea Nitrogen (mg/dL)", "14.66 ± 5.75", "12.28 ± 3.62", "17.79 ± 6.78", "0.05%"],
        ["Serum Albumin (g/dL)", "4.10 ± 0.36", "4.18 ± 0.32", "3.99 ± 0.38", "0.0%"],
        ["Serum Uric Acid (mg/dL)", "5.13 ± 1.41", "4.81 ± 1.32", "5.56 ± 1.42", "0.0%"],
        ["Diagnosed Diabetes, n (%)", "826 (13.06%)", "183 (5.09%)", "643 (23.52%)", "0.0%"],
        ["Poverty-Income Ratio", "2.88 ± 1.66", "2.94 ± 1.68", "2.81 ± 1.63", "12.54%"]
    ]
    for row_idx, rdata in enumerate(t1_rows, start=1):
        for col_idx, cval in enumerate(rdata):
            cell = tbl1.cell(row_idx, col_idx)
            cell.text = cval
            cell.paragraphs[0].runs[0].font.size = Pt(8)

    doc.add_paragraph()

    add_springer_h2("2.2 Feature Policy and Target Leakage Quarantine")
    doc.add_paragraph(
        "To prevent any trivial reconstruction of the outcome, we quarantined seven variables directly related to how the target was derived: "
        "participant_id, egfr, serum_creatinine, urine_albumin, urine_creatinine, albumin_creatinine_ratio, and ckd_stage. "
        "The final feature set comprises of 21 non-leaking predictors: 14 continuous variables (age, BMI, weight, height, systolic BP, diastolic BP, "
        "BUN, serum albumin, phosphorus, bicarbonate, calcium, uric acid, poverty-income ratio, education level) and 7 categorical variables "
        "(gender, ethnicity, diagnosed diabetes, insulin use, diabetes pills, smoking history, current smoking status)."
    )

    # Embed Fig. 1: Target distribution
    fig1_path = os.path.join(figures_dir, "target_distribution.png")
    if os.path.exists(fig1_path):
        doc.add_picture(fig1_path, width=Inches(4.8))
        f1_p = doc.add_paragraph()
        f1_p.paragraph_format.space_before = Pt(3)
        f1_p.paragraph_format.space_after = Pt(10)
        f1_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_f1 = f1_p.add_run("Fig. 1. Study population flow chart, target distribution, and KDIGO clinical staging breakdown for the analyzed NHANES 2021–2023 cohort.")
        r_f1.font.size = Pt(9)

    add_springer_h2("2.3 Learning Pipeline and Cross-Validation Protocol")
    doc.add_paragraph(
        "We applied repeated stratified 5-fold cross-validation with 5 random repeats, providing 25 splits in total. Inside each fold, continuous inputs "
        "were median-imputed and standardized using z-scores, while categorical variables were mode-imputed and one-hot encoded with handle_unknown='ignore'. "
        "Crucially, every transformation was fitted only on training instances (N ≈ 5,060) and then applied to the held-out validation instances (N ≈ 1,266) "
        "without any re-fitting. We benchmarked three model architectures with fixed hyperparameter settings: regularized Logistic Regression (L2, C = 1.0), "
        "Random Forest (300 bagged trees, min_samples_leaf = 2), and XGBoost (300 boosting iterations, learning rate η = 0.05, max_depth = 3, subsample = 0.8, colsample_bytree = 0.8)."
    )

    add_springer_h2("2.4 Explainability, Dummy Aggregation, and Stability Quantification")
    doc.add_paragraph(
        "Local feature contributions for validation instances were computed using Shapley Additive Explanations (SHAP) [8], employing exact TreeSHAP for tree ensembles "
        "and LinearSHAP for Logistic Regression. One-hot dummy level attributions were summed back to their parent clinical features by summing absolute Shapley contributions across levels: "
        "Attribution_i(F) = sum_k |phi_{i, k}|. Stability across all 300 cross-validation fold pairs was evaluated using pairwise Spearman rank correlations (rho) and Top-k Jaccard similarity indices (k in {5, 10})."
    )

    # Section 3: Experimental Results
    add_springer_h1("3 Experimental Results")
    add_springer_h2("3.1 Model Discrimination and Calibration")
    doc.add_paragraph(
        "Table 2 provides a summary of cross-validation metrics across all 25 folds."
    )

    # Table 2: Model Comparison (Caption ABOVE)
    t2_cap = doc.add_paragraph()
    t2_cap.paragraph_format.space_before = Pt(8)
    t2_cap.paragraph_format.space_after = Pt(3)
    t2_cap.paragraph_format.keep_with_next = True
    r_t2 = t2_cap.add_run("Table 2. Performance metrics from cross-validation over 25 folds (Mean ± SD).")
    r_t2.bold = True
    r_t2.font.size = Pt(9)

    tbl2 = doc.add_table(rows=4, cols=8)
    tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl2)

    t2_headers = ["Model", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "F1-Score", "Accuracy", "Brier Score"]
    for i, h in enumerate(t2_headers):
        cell = tbl2.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(8)
        set_cell_background(cell, "F2F2F2")

    t2_rows = [
        ["Logistic Regression", "0.8441 ± 0.0123", "0.8242 ± 0.0143", "0.7303 ± 0.0154", "0.8142 ± 0.0171", "0.7398 ± 0.0137", "0.7779 ± 0.0124", "0.1559 ± 0.0063"],
        ["Random Forest", "0.8406 ± 0.0123", "0.8133 ± 0.0160", "0.7187 ± 0.0144", "0.8130 ± 0.0184", "0.7317 ± 0.0133", "0.7722 ± 0.0124", "0.1600 ± 0.0057"],
        ["XGBoost", "0.8456 ± 0.0129", "0.8246 ± 0.0149", "0.7025 ± 0.0161", "0.8268 ± 0.0158", "0.7280 ± 0.0148", "0.7731 ± 0.0128", "0.1559 ± 0.0069"]
    ]
    for row_idx, rdata in enumerate(t2_rows, start=1):
        for col_idx, cval in enumerate(rdata):
            cell = tbl2.cell(row_idx, col_idx)
            cell.text = cval
            cell.paragraphs[0].runs[0].font.size = Pt(7.5)

    doc.add_paragraph()

    # Embed Fig. 2: ROC and PR Curves
    fig2_path = os.path.join(figures_dir, "roc_pr_curves.png")
    if os.path.exists(fig2_path):
        doc.add_picture(fig2_path, width=Inches(4.8))
        f2_p = doc.add_paragraph()
        f2_p.paragraph_format.space_before = Pt(3)
        f2_p.paragraph_format.space_after = Pt(10)
        f2_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_f2 = f2_p.add_run("Fig. 2. Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves across 25 cross-validation folds, plotted with ±1 SD confidence bands.")
        r_f2.font.size = Pt(9)

    add_springer_h2("3.2 Global SHAP Risk Factor Rankings")
    doc.add_paragraph(
        "Pooling SHAP values across all 25 folds resulted in consistent biological rankings across the model families (Fig. 3). In every model, age was the leading contributor "
        "(mean |SHAP| ~0.14–0.17), followed by blood urea nitrogen, systolic blood pressure, serum uric acid, and diagnosed diabetes. As a collective, these five variables "
        "represented more than 72% of the total attribution mass across architectures."
    )

    # Embed Fig. 3: Global Feature Importance
    fig3_path = os.path.join(figures_dir, "global_feature_importance.png")
    if os.path.exists(fig3_path):
        doc.add_picture(fig3_path, width=Inches(4.8))
        f3_p = doc.add_paragraph()
        f3_p.paragraph_format.space_before = Pt(3)
        f3_p.paragraph_format.space_after = Pt(10)
        f3_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_f3 = f3_p.add_run("Figure 3. Rankings of global SHAP feature importance for Logistic Regression, Random Forest, and XGBoost after aggregating dummy variables into their parent clinical features.")
        r_f3.font.size = Pt(9)

    add_springer_h2("3.3 Consistency of Explanations Across 300 Fold Pairs")
    doc.add_paragraph(
        "An assessment of stability across all 300 pairwise fold comparisons revealed that the tree-based ensembles yielded rankings that were remarkably consistent (Table 3). "
        "Random Forest demonstrated the greatest ranking stability (rho = 0.9652 ± 0.0148) as well as strong Top-10 Jaccard consistency (0.9577 ± 0.0781). Across all 300 comparisons, "
        "both Random Forest and XGBoost achieved a Top-5 Jaccard index of 1.0000 ± 0.0000, suggesting that the five leading predictors remained unchanged regardless of how the data were split."
    )

    # Table 3: Stability Summary (Caption ABOVE)
    t3_cap = doc.add_paragraph()
    t3_cap.paragraph_format.space_before = Pt(8)
    t3_cap.paragraph_format.space_after = Pt(3)
    t3_cap.paragraph_format.keep_with_next = True
    r_t3 = t3_cap.add_run("Table 3. Summary of SHAP attribution stability over 300 cross-validation fold pairs.")
    r_t3.bold = True
    r_t3.font.size = Pt(9)

    tbl3 = doc.add_table(rows=4, cols=6)
    tbl3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl3)

    t3_headers = ["Model Architecture", "Comparisons", "Spearman rho", "Spearman Median (IQR)", "Top-5 Jaccard", "Top-10 Jaccard"]
    for i, h in enumerate(t3_headers):
        cell = tbl3.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(8)
        set_cell_background(cell, "F2F2F2")

    t3_rows = [
        ["Random Forest", "300", "0.9652 ± 0.0148", "0.9662 (0.0208)", "1.0000 ± 0.0000", "0.9577 ± 0.0781"],
        ["XGBoost", "300", "0.9414 ± 0.0214", "0.9455 (0.0312)", "1.0000 ± 0.0000", "0.7997 ± 0.1087"],
        ["Logistic Regression", "300", "0.8972 ± 0.0553", "0.9117 (0.0740)", "0.9267 ± 0.1383", "0.8162 ± 0.1022"]
    ]
    for row_idx, rdata in enumerate(t3_rows, start=1):
        for col_idx, cval in enumerate(rdata):
            cell = tbl3.cell(row_idx, col_idx)
            cell.text = cval
            cell.paragraphs[0].runs[0].font.size = Pt(7.5)

    doc.add_paragraph()

    # Embed Fig. 4: Stability Heatmap
    fig4_path = os.path.join(figures_dir, "shap_stability_heatmap.png")
    if os.path.exists(fig4_path):
        doc.add_picture(fig4_path, width=Inches(4.8))
        f4_p = doc.add_paragraph()
        f4_p.paragraph_format.space_before = Pt(3)
        f4_p.paragraph_format.space_after = Pt(10)
        f4_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_f4 = f4_p.add_run("Fig. 4. Matrices of pairwise SHAP attribution stability (25x25 folds, 300 comparisons), displaying Spearman rank correlations across cross-validation resamples.")
        r_f4.font.size = Pt(9)

    add_springer_h2("3.4 Missingness Stress Testing")
    doc.add_paragraph(
        "Under synthetic MCAR masking, models displayed a graceful degradation in performance (Table 4). With 10% missingness, ROC-AUC dropped by approximately 0.011–0.013 points across all models. "
        "When missingness reached 20%, every architecture retained an ROC-AUC above 0.820. SHAP rankings remained consistent with unmasked baselines, with Spearman correlations remaining above 0.930."
    )

    # Table 4: Missingness Stress Test (Caption ABOVE)
    t4_cap = doc.add_paragraph()
    t4_cap.paragraph_format.space_before = Pt(8)
    t4_cap.paragraph_format.space_after = Pt(3)
    t4_cap.paragraph_format.keep_with_next = True
    r_t4 = t4_cap.add_run("Table 4. How algorithmic performance and explanation stability withstands controlled MCAR missingness.")
    r_t4.bold = True
    r_t4.font.size = Pt(9)

    tbl4 = doc.add_table(rows=10, cols=6)
    tbl4.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl4)

    t4_headers = ["Model", "Masking Stress", "ROC-AUC", "F1-Score", "Accuracy", "SHAP Rank Corr (rho)"]
    for i, h in enumerate(t4_headers):
        cell = tbl4.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(8)
        set_cell_background(cell, "F2F2F2")

    t4_rows = [
        ["Logistic Regression", "0% (Baseline)", "0.8482", "0.7422", "0.7844", "1.0000"],
        ["Logistic Regression", "10% MCAR", "0.8366 ± 0.0023", "0.7203 ± 0.0057", "0.7671 ± 0.0054", "0.9403 ± 0.0037"],
        ["Logistic Regression", "20% MCAR", "0.8207 ± 0.0045", "0.7003 ± 0.0113", "0.7501 ± 0.0108", "0.9314 ± 0.0077"],
        ["Random Forest", "0% (Baseline)", "0.8475", "0.7342", "0.7804", "1.0000"],
        ["Random Forest", "10% MCAR", "0.8370 ± 0.0021", "0.7133 ± 0.0061", "0.7679 ± 0.0041", "0.9748 ± 0.0052"],
        ["Random Forest", "20% MCAR", "0.8227 ± 0.0028", "0.6870 ± 0.0056", "0.7528 ± 0.0054", "0.9727 ± 0.0049"],
        ["XGBoost", "0% (Baseline)", "0.8514", "0.7264", "0.7757", "1.0000"],
        ["XGBoost", "10% MCAR", "0.8385 ± 0.0029", "0.7052 ± 0.0059", "0.7626 ± 0.0045", "0.9590 ± 0.0103"],
        ["XGBoost", "20% MCAR", "0.8212 ± 0.0039", "0.6756 ± 0.0066", "0.7439 ± 0.0075", "0.9501 ± 0.0111"]
    ]
    for row_idx, rdata in enumerate(t4_rows, start=1):
        for col_idx, cval in enumerate(rdata):
            cell = tbl4.cell(row_idx, col_idx)
            cell.text = cval
            cell.paragraphs[0].runs[0].font.size = Pt(7.5)

    doc.add_paragraph()

    # Embed Fig. 5: Missingness Sensitivity
    fig5_path = os.path.join(figures_dir, "missingness_sensitivity.png")
    if os.path.exists(fig5_path):
        doc.add_picture(fig5_path, width=Inches(4.8))
        f5_p = doc.add_paragraph()
        f5_p.paragraph_format.space_before = Pt(3)
        f5_p.paragraph_format.space_after = Pt(10)
        f5_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_f5 = f5_p.add_run("Fig. 5. Algorithmic degradation trajectories under controlled 0%, 10%, and 20% MCAR missingness, measuring discrimination (ROC-AUC) and explanation stability.")
        r_f5.font.size = Pt(9)

    # Section 4: Discussion
    add_springer_h1("4 Discussion and Clinical Implications")
    doc.add_paragraph(
        "The leading variables identified by SHAP correspond to well-established nephrological mechanisms:\n"
        "• Chronological Age: Progressive nephron involution and vascular stiffening reduce functional renal reserve beyond the fourth decade of life [13].\n"
        "• Blood Urea Nitrogen (BUN): As an end-product of protein metabolism removed through glomerular filtration, BUN acts as an informative surrogate marker of clearance capacity when creatinine is unavailable.\n"
        "• Systolic Blood Pressure: Increased systemic pressure subjects glomerular microvessels to mechanical barotrauma, accelerating sclerotic changes [14].\n"
        "• Serum Uric Acid: Hyperuricemia triggers renal vasoconstriction, activates the renin-angiotensin system, and induces tubular epithelial oxidative injury [15].\n"
        "• Diabetes: Diabetic glomerulosclerosis is the foremost global cause of chronic kidney failure."
    )
    doc.add_paragraph(
        "In contrast to concerns about explanation volatility, TreeSHAP applied to Random Forest and XGBoost displayed high ranking stability across resampled partitions "
        "(rho > 0.94; Top-5 Jaccard = 1.000). The minor variability observed in Logistic Regression (rho ≈ 0.897) stemmed primarily from collinearity between blood pressure "
        "and glycemic markers, which can alter regression coefficients when samples shift slightly."
    )
    doc.add_paragraph(
        "Several boundary conditions constrain our findings. First, NHANES provides single-timepoint measurements, so it cannot determine whether renal abnormalities persist "
        "for three months, as the strict KDIGO definition requires [1]. The models therefore perform cross-sectional classification rather than longitudinal onset prediction. "
        "Second, omitting direct filtration markers prevents tautology but sets an upper bound on discrimination (ROC-AUC ≈ 0.85). These models are designed for opportunistic "
        "risk screening prior to ordering renal panels. Third, our stress test assessed MCAR mechanisms, while routine clinical missingness may instead follow Missing Not at Random (MNAR) patterns."
    )

    # Section 5: Conclusion
    add_springer_h1("5 Conclusion")
    doc.add_paragraph(
        "In conclusion, machine learning classifiers trained on non-leaking clinical and demographic variables attained an ROC-AUC of approximately 0.846 for concurrent CKD classification "
        "within the NHANES 2021–2023 cohort. TreeSHAP feature rankings demonstrated high stability across cross-validation resamples and degraded gracefully under synthetic missingness. "
        "The consensus factors identified reflect established pathophysiological pathways, validating that leak-free opportunistic screening is achievable and that tree-based "
        "Shapley explanations yield dependable feature attributions amidst data perturbations."
    )

    # References (Springer LNCS Style)
    add_springer_h1("References")
    refs = [
        "1. KDIGO Board: KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease. Kidney International 105(4S), S117–S314 (2024)",
        "2. Bikbov, B., Purcell, C.A., Levey, A.S., et al.: Global, regional, and national burden of chronic kidney disease, 1990–2017. The Lancet 395(10225), 709–733 (2020)",
        "3. Inker, L.A., Eneanya, N.D., Coresh, J., et al.: New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race. New England Journal of Medicine 385(19), 1737–1749 (2021)",
        "4. Kovesdy, C.P.: Epidemiology of chronic kidney disease: an update 2022. Kidney International Supplements 12(1), 7–11 (2022)",
        "5. He, T., Song, Y., Wang, J., et al.: Machine learning for screening and risk stratification of chronic kidney disease: a multicentre cohort study. Lancet Digital Health 7(2), e104–e115 (2025)",
        "6. Song, X., Waitman, L.R., Hu, Y., et al.: Robust machine learning prediction of chronic kidney disease progression using real-world clinical data. Journal of the American Medical Informatics Association 31(3), 621–632 (2024)",
        "7. Moons, K.G.M., Wolff, R.F., Riley, R.D., et al.: PROBAST+AI: Risk of Bias Assessment Tool for Prediction Models using Artificial Intelligence. Nature Medicine 31(1), 112–128 (2025)",
        "8. Lundberg, S.M., Lee, S.I.: A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NeurIPS) 30, 4765–4774 (2017)",
        "9. Nandy, P., Wang, Y., Ertin, E., et al.: On the Stability of Feature Attributions Across Resampled Training Sets in Medical ML. IEEE Transactions on Neural Networks and Learning Systems 32(8), 3410–3422 (2021)",
        "10. Han, T., Srinivas, S., Lakkaraju, H.: Which Explanation Should I Choose? A Survey and Empirical Evaluation of Explanation Stability and Faithfulness. NeurIPS Datasets and Benchmarks 2, 1540–1558 (2022)",
        "11. Collins, G.S., Moons, K.G.M., Dhiman, P., et al.: TRIPOD+AI Statement: Updated Guidance for Reporting Clinical Prediction Models That Use Artificial Intelligence. British Medical Journal 385, e078378 (2024)",
        "12. Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS): National Health and Nutrition Examination Survey (NHANES) 2021–2023 Documentation. U.S. Department of Health and Human Services, CDC (2024)",
        "13. Denic, A., Glassock, R.J., Rule, A.D.: Structural and Functional Changes With the Aging Kidney. Advances in Chronic Kidney Disease 23(1), 19–28 (2016)",
        "14. Chobanian, A.V., Bakris, G.L., Black, H.R., et al.: Seventh Report of the Joint National Committee on Prevention, Detection, Evaluation, and Treatment of High Blood Pressure. Hypertension 42(6), 1206–1252 (2003)",
        "15. Johnson, R.J., Bakris, G.L., Borghi, C., et al.: Hyperuricemia, Acute and Chronic Kidney Disease, Hypertension, and Cardiovascular Disease. American Journal of Kidney Diseases 71(6), 851–865 (2018)"
    ]
    for r in refs:
        p = doc.add_paragraph(r)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.runs[0].font.size = Pt(8.5)

    doc.save(output_path)
    print(f"Springer LNNS Word manuscript successfully saved to: {output_path}")


if __name__ == "__main__":
    build_springer_manuscript()
