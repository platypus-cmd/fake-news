# Scientific Accuracy and Consistency Audit Report

**Project:** Replication of Ahmad et al., *"Fake News Detection Using Machine Learning Ensemble Methods"* (*Complexity*, 2020, Article ID 8885861)  
**Authors:** Neil Kaushik, Sameer Saini  
**Date:** November 6, 2026  
**Status:** Audit Completed — Experimental Data Locked & Verified

---

## Executive Summary

A targeted scientific accuracy and consistency audit was conducted across the replication research paper source files (`results/fake_news_replication_paper.md`, `results/fake_news_replication_paper.html`, and `results/fake_news_replication_paper_final.html`). All eight designated audit items have been resolved without altering experimental source code, modifying datasets, changing model configurations, or re-running experiments. All measured numerical results remain strictly locked against the benchmark evaluation logs (`results/evaluation_metrics_tuned.csv` and `results/paper_comparison_tuned.csv`).

---

## Audit Item Resolutions

### 1. 93-Feature Category Accounting
- **Issue:** The feature category breakdown in Section V.B previously summed to 87 features ($6 + 6 + 11 + 11 + 5 + 10 + 7 + 7 + 8 + 16 = 87$), omitting 6 features while claiming a 93-feature representation.
- **Verification:** Inspected `src/liwc_features.py` to extract the canonical `FEATURE_NAMES` manifest (length: exactly 93). Reconciled the 6 missing variables: indices 87–92 are explicitly defined in the extractor as fixed constant-zero dimensionality alignment features (`pad_88` through `pad_93`) ensuring structural vector parity with LIWC2015's 93-column specification.
- **Correction Applied:** Reconciled Section V.B in both Markdown and HTML to explicitly account for all 11 functional categories:
  1. Summary & Text Statistics: **6 features** (`wc`, `wps`, `sixltr`, `dic`, `numerals`, `avg_word_len`)
  2. Punctuation Marks: **6 features** (`period`, `comma`, `colon`, `excl`, `qmark`, `dash`)
  3. Function Words & Pronouns: **11 features** (`pronoun`, `ppron`, `i`, `we`, `you`, `shehe`, `they`, `ipron`, `article`, `prep`, `auxverb`)
  4. Social and Affective Processes: **11 features** (`social`, `family`, `friend`, `female`, `male`, `affect`, `posemo`, `negemo`, `anx`, `anger`, `sad`)
  5. Cognitive Processes: **5 features** (`cogproc`, `insight`, `cause`, `discrep`, `tentat`)
  6. Perceptual and Biological Terms: **10 features** (`percept`, `see`, `hear`, `feel`, `bio`, `body`, `health`, `sexual`, `ingest`, `drives`)
  7. Time and Space Orientation: **7 features** (`focuspast`, `focuspresent`, `focusfuture`, `relativ`, `motion`, `space`, `time`)
  8. Personal Concerns: **7 features** (`work`, `achieve`, `leisure`, `home`, `money`, `relig`, `death`)
  9. Grammatical Markers & POS Ratios: **8 features** (`verb`, `adj`, `compare`, `interrog`, `number_wrd`, `quant`, `conjunctions`, `negate`)
  10. Informal Language & Readability: **16 features** (`informal`, `swear`, `netspeak`, `assent`, `nonflu`, `filler`, `unique_words_ratio`, `sentence_count`, `avg_sentence_len`, `stopword_ratio`, `sentiment_pos`, `sentiment_neg`, `sentiment_compound`, `has_url`, `allcaps_ratio`, `title_case_ratio`)
  11. Dimensionality Alignment Padding: **6 features** (`pad_88`, `pad_89`, `pad_90`, `pad_91`, `pad_92`, `pad_93`)
- **Total Accounting:** $6 + 6 + 11 + 11 + 5 + 10 + 7 + 7 + 8 + 16 + 6 = 93$ features.

---

### 2. Ensemble Performance Claims
- **Issue:** The Abstract and Conclusion broadly claimed that ensemble methods "consistently outperform single classifiers." This was factually overbroad because AdaBoost's mean accuracy (0.8909) is lower than three individual classifiers: Multilayer Perceptron (0.9122), Linear SVM (0.8985), and Logistic Regression (0.8916). Moreover, MLP outperformed AdaBoost across all four individual datasets.
- **Correction Applied:** Rewrote claims across the Abstract, Section VII.B, Section IX.C, and Section XI.B. Specifically:
  - Clarified that *tree-based ensembles* (XGBoost at 0.9327 mean accuracy, Random Forest at 0.9291, and Bagging at 0.9233) consistently led overall performance.
  - Explicitly documented that ensemble superiority was not universal: AdaBoost (0.8909) underperformed individual models (MLP, Linear SVM, and Logistic Regression), indicating that shallow decision stumps ($d=1$) struggle to model intricate linguistic feature interactions as effectively as deeper tree structures or continuous neural representations.

---

### 3. Precision Interpretation (Label Semantics)
- **Issue:** Section IX.C previously claimed that Random Forest's high precision (mean 0.9384) made it "well-suited for settings prioritizing minimal false-positive flagging."
- **Verification:** Verified label schema in `src/config.py`: `LABEL_FAKE = 0`, `LABEL_TRUTH = 1`. In scikit-learn binary classification (`average="binary"`), precision defaults to `pos_label=1` (Truthful).
- **Correction Applied:** Corrected the scientific explanation in Section IX.C:
  - Precision measures $\text{TP} / (\text{TP} + \text{FP})$, where $\text{TP}$ is Truthful predicted Truthful, and $\text{FP}$ is Fake predicted Truthful.
  - A high positive-class precision (RF: 0.9384, XGB: 0.9333) demonstrates high fidelity when certifying legitimate journalism—meaning Random Forest rarely misclassifies fake articles as truthful (low fake-news contamination in the trusted news stream).
  - Clarified that this metric does *not* directly measure the avoidance of falsely accusing legitimate news of being fake (which corresponds to precision on class 0, or true-negative specificity).

---

### 4. LIWC Terminology and Equivalence Distinctions
- **Issue:** Section V.B previously described the proxy as extracting an "identical 93-dimensional vector," which could mislead readers into assuming numeric equivalence with the commercial LIWC2015 dictionary output.
- **Correction Applied:** Replaced misleading phrasing with precise technical language:
  > *"The extractor parses raw text and extracts a correspondingly structured 93-dimensional feature vector—matching the exact dimensionality and broad psychological categories of LIWC2015, while relying on public lexicons and rule-based extractors rather than the proprietary LIWC2015 dictionary stems."*

---

### 5. Author Information and Institutional Placeholders
- **Issue:** Ensure author details and affiliations contain no fabricated information and remain clearly marked for manual completion.
- **Status Verified:** Confirmed that the author blocks in both Markdown and HTML preserve clean, un-fabricated placeholders:
  - Author 1: Neil Kaushik | Department of Computer Science and Engineering | `[College / University Affiliation]` | `*[College Email]*`
  - Author 2: Sameer Saini | Department of Computer Science and Engineering | `[College / University Affiliation]` | `*[College Email]*`

---

### 6. Reference and Citation Consistency
- **Audit Findings:**
  1. **Primary Reference Linked:** Ahmad et al. (2020) was previously cited by name in text without an explicit numeric bracket citation; attached `[40]` in Abstract, Introduction, Section II.C, Section III, Section V.B, Section VIII, and Section XI.B.
  2. **LIWC Citation Corrected:** In Section II.A, Pennebaker et al. was erroneously cited as `[10]` (Conroy et al.); corrected to `[41]` (Pennebaker et al., 2015, *"The development and psychometric properties of LIWC2015"*).
  3. **Math Mode Bracket Distinction:** Verified that `$[0, 1]$` in Section V.C is KaTeX interval notation, not a broken citation.
  4. **Uncited Historical References Flagged:** References [2], [4]–[9], [11], [16], [18]–[20], [29]–[31], and [36] are listed in the 41-reference bibliography because they were directly inherited from Ahmad et al.'s original 39-reference literature review.
  5. **Bibliographic Completeness Flagged:** Reference [11] (*MisinfoText*) lacks a formal publication venue/URL; Reference [31] pertains to Arabic speech mispronunciation detection (inherited from Ahmad et al.'s survey of neural network applications). These remain untouched to preserve bibliographic fidelity to the replicated study without fabricating details.

---

### 7. Layout and Table Readability (PDF Pages 5–7)
- **Problem Observed:** In the rendered PDF, Table III (40 rows $\times$ 9 columns) and Table VI (40 rows $\times$ 7 columns) were constrained within a narrow single column of a two-column layout. When printed with `break-inside: avoid;`, the tables either caused multi-page overflow or severe vertical text wrapping (clipping `Train Time (s)`).
- **Improvements Implemented:**
  - **Font Size & Line Height:** Reduced table font size from `7.6pt` to `6.8pt` with `line-height: 1.15;`, utilizing `font-variant-numeric: tabular-nums;` for precise vertical decimal alignment.
  - **Pagination Control:** Set `.table-wrapper { break-inside: auto; }` and `table.ieee-table tr { break-inside: avoid; page-break-inside: avoid; }`, allowing clean page breaks between rows while preventing orphan rows.
  - **Repeating Headers:** Added `table.ieee-table thead { display: table-header-group; }` so column headers repeat automatically when a table breaks across a column or page.
  - **Dataset Separation:** Injected distinct visual separator borders (`border-top: 1px solid #777;`) between DS1, DS2, DS3, and DS4 blocks in Table III and Table VI, allowing immediate visual orientation without hunting across 40 rows.
  - **PDF Verification:** Re-rendered the paper using Edge headless PDF export; confirmed total length compacted cleanly to 10 pages with complete visibility of all 9 columns in Table III and all 7 columns in Table VI.

---

### 8. Final Numerical Audit Against CSV Files
All numerical claims across the Abstract, Table I, Table II, Table III, Table IV, Table V, Table VI, Section IX, and Section XI were audited via automated Python cross-verification against the locked CSV files:

| Verification Target | Source Data File | Check Status | Notes |
| :--- | :--- | :---: | :--- |
| **Cleaned Article Counts** | `data/processed/*.csv` | **100% Concordance** | DS1: 44,182; DS2: 20,408; DS3: 3,990; DS4: 68,580 |
| **Table III Metrics (40 rows)** | `results/evaluation_metrics_tuned.csv` | **100% Concordance** | All 40 accuracy, precision, recall, F1, and time values match to 4 decimal places |
| **Table IV Means (10 models)** | Calculated from `evaluation_metrics_tuned.csv` | **100% Concordance** | XGB: 0.9327, RF: 0.9291, Bagging: 0.9233, VC1: 0.9159, MLP: 0.9122, VC2: 0.8997, LSVM: 0.8985, LR: 0.8916, AdaBoost: 0.8909, KNN: 0.8653 |
| **Table V Tuning Gains (8 rows)** | Baseline vs. Tuned CSVs | **100% Concordance** | LR mean gain: +0.0164, AdaBoost: +0.0109, VC1: +0.0104, VC2: +0.0092, LSVM: +0.0079, KNN: +0.0044, RF: +0.0005, MLP/XGB: 0.0000 |
| **Table VI Paper Comparisons** | `results/paper_comparison_tuned.csv` | **100% Concordance** | All 40 paper accuracies and differences match Table 2 of Ahmad et al. (2020) |
| **GridSearch Workload** | Experimental execution logs | **100% Concordance** | 22 parameter combinations $\times$ 10 folds $\times$ 4 datasets = 880 CV fits |

---

## Remaining Unresolved Items (For Manual Author Action)

1. **Author Institutional Affiliation:** Replace `[College / University Affiliation]` with your actual department, college, or university name in `results/fake_news_replication_paper.md` and `results/fake_news_replication_paper.html`.
2. **Author Email Addresses:** Replace `*[College Email]*` with your institutional or academic email addresses.
