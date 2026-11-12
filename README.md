# Fake News Detection Using Machine Learning Ensemble Methods: An Independent Replication and Empirical Evaluation

An independent scientific replication and empirical evaluation of the research paper:
> **H. Ahmad, M. U. Asghar, M. Z. Asghar, A. Khan, and A. H. Mosavi**, *"Fake News Detection Using Machine Learning Ensemble Methods,"* *Complexity*, vol. 2020, Article ID 8885861, 11 pages, 2020. [DOI: 10.1155/2020/8885861](https://doi.org/10.1155/2020/8885861)

**Team Members:** Neil Kaushik & Sameer Saini  
**Affiliation:** Department of Computer Science and Engineering  

---

## 📌 Project Overview

This project independently replicates and evaluates the psycholinguistic fake news detection pipeline proposed by Ahmad et al. (2020). Traditional content-based fake news detection relying on TF-IDF or keyword $n$-grams often suffers from domain shift when deployed across varying topics. Ahmad et al. proposed using **psycholinguistic stylometry** (extracting writing style, emotional tone, and structural characteristics via LIWC2015) paired with **ensemble machine learning models**.

Because commercial LIWC2015 is closed-source and proprietary, our replication engineers a transparent, fully reproducible **open-source linguistic feature extractor** capturing 87 active linguistic measurements (lexical, affective, cognitive, grammatical, and structural) supplemented by 6 constant-zero padding features to construct an aligned 93-dimensional feature space.

---

## 🔬 Key Experimental Results

Ten classifiers (4 individual baselines + 6 ensemble architectures) were evaluated across four benchmark datasets totaling **68,580 articles** using a 70/30 stratified train-test split and 10-fold cross-validated grid search (880 total CV fits):

| Rank | Model Name | Category | Cross-Dataset Mean Accuracy | Mean Precision (Truthful) | Mean Recall | Mean F1-Score |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 **1** | **XGBoost (XGB)** | Ensemble | **93.27%** | 93.33% | 93.00% | **93.16%** |
| 🥈 **2** | **Random Forest (RF)** | Ensemble | **92.91%** | **93.84%** | 91.61% | **92.69%** |
| 🥉 **3** | **Bagging (Decision Trees)** | Ensemble | **92.33%** | 93.01% | 91.28% | **92.12%** |
| **4** | **Voting Classifier 1 (RF+LR+KNN)** | Ensemble | **91.59%** | 90.66% | **92.40%** | **91.50%** |
| **5** | **Multilayer Perceptron (MLP)** | Individual | **91.22%** | 90.38% | 91.96% | **91.12%** |
| **6** | **Voting Classifier 2 (LR+LSVM+DT)**| Ensemble | **89.97%** | 89.26% | 90.48% | **89.83%** |
| **7** | **Linear SVM (LSVM)** | Individual | **89.85%** | 89.15% | 90.35% | **89.70%** |
| **8** | **Logistic Regression (LR)** | Individual | **89.16%** | 88.33% | 89.78% | **89.01%** |
| **9** | **AdaBoost** | Ensemble | **89.09%** | 88.61% | 89.31% | **88.93%** |
| **10**| **K-Nearest Neighbors (KNN)** | Individual | **86.53%** | 83.07% | 91.54% | **87.06%** |

### Key Discoveries:
1. **Tree Ensembles Lead:** XGBoost achieved the highest cross-dataset mean accuracy (93.27%), and Random Forest achieved top performance on the mixed DS4 corpus (91.26%).
2. **Ensemble Superiority is Not Universal:** AdaBoost (89.09%) underperformed individual models like Multilayer Perceptron (91.22%) and Linear SVM (89.85%), demonstrating that shallow boosting stumps ($d=1$) struggle with complex multi-feature interactions.
3. **Open-Source Proxy Viability:** Highly competitive classification performance (>91%–96%) is achievable using accessible, open-source linguistic approximations without relying on proprietary software.

---

## 📊 Datasets Evaluated

| Dataset | Source / Description | Fake ($y=0$) | Truthful ($y=1$) | Total Cleaned Articles | Fake Ratio (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DS1** | ISOT Research Dataset (University of Victoria) | 22,766 | 21,416 | 44,182 | 51.53% |
| **DS2** | Kaggle Fake News Competition | 10,022 | 10,386 | 20,408 | 49.11% |
| **DS3** | Kaggle Fake News Detection (Jruvika) | 2,122 | 1,868 | 3,990 | 53.18% |
| **DS4** | Combined Master Corpus (DS1 + DS2 + DS3) | 34,910 | 33,670 | 68,580 | 50.90% |

*Label Mapping:* $0 = \text{Fake / Unreliable}$, $1 = \text{Truthful / Reliable}$.

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/                 # Untouched raw datasets (ISOT, Kaggle, Jruvika)
│   └── processed/           # Cleaned CSVs & scaled feature matrices
├── models/                  # Serialized MinMaxScaler (.joblib) objects
├── notebooks/               # Exploratory data analysis notebooks
├── results/                 # Evaluation logs, paper comparisons, and plots
│   ├── evaluation_metrics_tuned.csv
│   ├── paper_comparison_tuned.csv
│   ├── best_params.csv
│   ├── fake_news_replication_paper.pdf  # 9-page IEEE format paper
│   └── plots/               # Evaluation charts (accuracy, tuning, comparisons)
├── src/
│   ├── config.py            # Central parameters & hyperparameter search grids
│   ├── data_preprocessing.py# Text cleaning, filtering, and dataset concatenation
│   ├── liwc_features.py     # 87 active + 6 zero padding feature extractor
│   ├── feature_pipeline.py  # Stratified splitting, MinMaxScaler, and caching
│   ├── models.py            # Model factory & 10-fold GridSearchCV wrapper
│   └── train_models.py      # Execution engine for training and evaluation
├── requirements.txt         # Package dependencies
└── README.md
```

---

## 🚀 Reproduction & Usage

### 1. Installation & Environment Setup
```bash
git clone git@github.com:platypus-cmd/fake-news.git
cd fake-news

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline
```bash
# Step 1: Preprocess raw datasets and generate DS1-DS4 cleaned CSVs
python src/data_preprocessing.py

# Step 2: Extract 93 linguistic features, apply 70/30 stratified split, and scale
python -m src.feature_pipeline

# Step 3: Run 10-fold cross-validated grid search and test-set evaluation
python -m src.train_models
```

---

## 📄 Publications & Artifacts
The full 9-page academic replication manuscript formatted in IEEE double-column layout is available in:
- PDF: [`results/fake_news_replication_paper.pdf`](results/fake_news_replication_paper.pdf)
- HTML: [`results/fake_news_replication_paper.html`](results/fake_news_replication_paper.html)
- Markdown: [`results/fake_news_replication_paper.md`](results/fake_news_replication_paper.md)
