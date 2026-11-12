# Fake News Detection Using Machine Learning Ensemble Methods: An Independent Replication and Empirical Evaluation

**Neil Kaushik**  
Department of Computer Science and Engineering  
[College / University Affiliation]  
*[College Email]*

**Sameer Saini**  
Department of Computer Science and Engineering  
[College / University Affiliation]  
*[College Email]*

---

### Abstract
The proliferation of digital disinformation presents severe challenges to democratic institutions, public health communication, and social stability. In 2020, Ahmad et al. [1] published an empirical study (*Complexity*, Article ID 8885861) asserting that ensemble machine learning methods combined with psycholinguistic feature extraction achieve superior classification accuracy over individual classifiers across diverse news domains. This paper presents an independent, end-to-end replication and empirical evaluation of that study. We evaluate ten machine learning models—four individual baseline classifiers (Logistic Regression, Linear Support Vector Machines, Multilayer Perceptron, and K-Nearest Neighbors) and six ensemble architectures (Random Forest, Bagging with Decision Trees, AdaBoost, XGBoost, and two hard-voting ensembles)—across four benchmark datasets comprising 68,580 articles (DS1: ISOT, DS2: Kaggle Fake News, DS3: Jruvika, and DS4: Combined master corpus). Because the commercial LIWC2015 dictionary utilized in the original study is proprietary and closed-source, our replication implements a transparent open-source LIWC-style approximation comprising 87 active linguistic measurements plus 6 constant-zero padding dimensions (yielding a 93-dimensional feature vector aligned structurally with LIWC2015's schema). Models were trained using a 70/30 stratified train-test split, scaled using MinMaxScaler normalization fitted across the 70% training partition (isolating the held-out test set while allowing internal validation folds to share scaling bounds during hyperparameter tuning), and optimized through 10-fold cross-validated grid search (880 total cross-validation fits across the four corpora). Our experimental findings broadly support the central conclusion of Ahmad et al. regarding tree-based ensemble efficacy: advanced tree ensembles (XGBoost, Random Forest, and Bagging) consistently achieved top-tier performance, with XGBoost achieving the highest cross-dataset mean accuracy (0.9327, compared to 0.9375 reported in the original paper) and Random Forest achieving top performance on the heterogeneous DS4 corpus (0.9126 accuracy). However, ensemble superiority was not universal across all ensemble architectures: AdaBoost (mean accuracy 0.8909) underperformed individual classifiers including Multilayer Perceptron (0.9122), Linear SVM (0.8985), and Logistic Regression (0.8916). Voting Ensemble 1 (RF + LR + KNN) achieved a mean accuracy of 0.9159, closely tracking the paper’s reported 0.9175. On DS2, where the original paper documented substantial performance drops among individual classifiers (0.28 to 0.37), our standardized preprocessing and scaling pipeline achieved stable base-learner accuracies between 0.8052 and 0.8685. This paper analyzes the implications of linguistic proxies, documents methodological safeguards against data leakage, and provides complete reproduction artifacts.

**Keywords**—Fake news detection, ensemble learning, natural language processing, linguistic inquiry, machine learning, replication study, XGBoost, Random Forest, cross-validation.

---

## I. Introduction

The rapid decentralization of media consumption and the emergence of algorithmic social feeds have accelerated the transmission of fabricated news stories. Disinformation campaigns during major geopolitical events, elections, and public health emergencies have highlighted the urgent need for robust, automated computational mechanisms capable of verifying informational veracity [2], [3], [4]. 

Traditional content-based natural language processing approaches rely heavily on bag-of-words or high-dimensional $n$-gram term frequency-inverse document frequency (TF-IDF) representations [5]. While TF-IDF models often capture domain-specific vocabulary effectively, they are susceptible to domain shift; an algorithm trained on political commentary frequently fails when deployed on healthcare or financial reporting [6]. Furthermore, deceptive text is deliberately crafted to mirror topical keywords of legitimate journalism while employing distinct rhetorical, affective, and psychological linguistic patterns [7], [8].

To address this challenge, Ahmad et al. (2020) [1] proposed a framework pairing psychological linguistic analysis with ensemble machine learning architectures [9], [10]. Using Linguistic Inquiry and Word Count (LIWC2015) [11], the authors extracted 93 continuous stylistic features per article and evaluated a spectrum of individual and ensemble classifiers across three distinct datasets and a unified master corpus. Their findings asserted that tree-based ensemble methods (notably XGBoost, Random Forest, and Bagging) and majority voting ensembles achieve superior generalization and accuracy compared to standard linear and distance-based baselines.

Independent replication is a foundational pillar of scientific inquiry. Computational studies frequently suffer from reproducibility bottlenecks arising from closed-source tools, ambiguous preprocessing procedures, undocumented data splits, or unstated data leakage. The primary objective of this paper is to provide an exhaustive, independent replication of Ahmad et al. (2020) [1]. 

Our study makes the following contributions:
1. **Open-Source Linguistic Implementation**: Because the commercial LIWC2015 software is proprietary and its dictionary cannot be redistributed, we engineer an open-source linguistic approximation comprising 87 active linguistic measurements plus 6 constant-zero padding dimensions (yielding a 93-dimensional feature vector aligned structurally with LIWC2015's schema) using public resources and rule-based extractors.
2. **Methodological Rigor**: We establish an audited pipeline enforcing strict held-out test set isolation, documenting that fitting MinMaxScaler across the full training partition permits internal validation folds to share scaling bounds during cross-validation, and applying 10-fold cross-validated grid search on training splits only.
3. **Comprehensive Empirical Evaluation**: We evaluate all ten classifiers across all four benchmark datasets (DS1, DS2, DS3, and the combined DS4), recording Accuracy, Precision, Recall, F1-score, and wall-clock training durations.
4. **Discrepancy Analysis**: We compare our experimental findings directly against Table 2 of Ahmad et al., examining areas of concordance (e.g., DS1, DS3, DS4, and Voting Ensembles) and documenting the divergence observed on DS2 individual baselines.

---

## II. Related Work

Automated deception detection in text has evolved through several methodological phases, spanning stylometry, semantic modeling, and deep representation learning.

### A. Stylometric and Linguistic Analysis
Early deception detection research established that deceptive communicators exhibit measurable deviations in pronoun utilization, emotional intensity, cognitive complexity, and structural punctuation [7]. Rubin et al. [8] demonstrated that satirical and deceptive articles employ higher rates of informal punctuation, exaggerative adjectives, and affective phrasing compared to legitimate news. Pérez-Rosas et al. [12] introduced standardized datasets and benchmarked Linear Support Vector Machines (LinearSVC) over combinations of lexical and psycholinguistic features, reporting accuracies between 70% and 78%. Pennebaker et al. developed the Linguistic Inquiry and Word Count (LIWC) framework, mapping words into psychologically validated categories [11]. LIWC features have since been widely adopted to capture behavioral cues without overfitting to domain-specific entities.

### B. Machine Learning Classifiers
Prior benchmark studies relied heavily on traditional supervised learning algorithms. Conroy et al. [7] surveyed baseline text classification methods, highlighting Logistic Regression (LR) and Support Vector Machines (SVM) as strong linear baselines for binary veracity classification. Logistic Regression provides calibrated posterior class probabilities [13], while Linear SVM constructs a maximum-margin separating hyperplane [14]. Non-linear baselines, such as Multilayer Perceptrons (MLP) and K-Nearest Neighbors (KNN), have also been explored, though KNN often incurs high computational latency during inference across large document corpora [6].

### C. Ensemble Learning Methods
Ensemble learning combines multiple base estimators to reduce variance, reduce bias, or improve predictive accuracy [9]. Decision tree classifiers, formalized by Breiman et al. via Classification and Regression Trees (CART) [15], provide flexible non-linear representations but are known to suffer from high variance. To stabilize decision tree estimates, Breiman introduced Bagging (Bootstrap Aggregating) [16], which trains an ensemble of trees on bootstrap replicates of the training data and averages their predictions. Breiman subsequently introduced Random Forests (RF) [17], injecting randomized feature subspace selection at each candidate split to reduce inter-tree correlation and yield strong variance reduction [18]. Boosting algorithms, including AdaBoost [19], [20] and Extreme Gradient Boosting (XGBoost) by Chen and Guestrin [21], iteratively train weak decision trees to minimize empirical loss functions by upweighting previously misclassified instances. Voting ensembles combine heterogeneous models through hard or soft consensus voting [10], [22]. Ahmad et al. (2020) [1] demonstrated that these ensemble paradigms provide resilient boundaries against domain variation in fake news corpora.

---

## III. Original Study and Replication Objective

The reference investigation for this replication is:
> H. Ahmad, M. U. Asghar, M. Z. Asghar, A. Khan, and A. H. Mosavi, "Fake News Detection Using Machine Learning Ensemble Methods," *Complexity*, vol. 2020, Article ID 8885861, 11 pages, 2020 [1].

The original study evaluated ten machine learning models across four dataset configurations using 93 features extracted with the commercial LIWC2015 application. The authors reported that ensemble models—specifically XGBoost, Random Forest, and Bagging—consistently outperformed individual learners, achieving overall accuracy rates up to 99% on DS1 and 90% on DS4.

### Replication Objective
Our replication aims to:
1. Reconstruct the complete data preprocessing, feature extraction, scaling, and training pipelines from first principles.
2. Substitute the proprietary LIWC2015 feature extractor with a transparent, fully reproducible open-source linguistic approximation comprising 87 active linguistic measurements and 6 constant-zero padding dimensions (yielding a 93-dimensional vector aligned with LIWC2015's schema).
3. Train and tune all ten classifiers using 10-fold cross-validated grid search under strict train/test isolation.
4. Evaluate whether the empirical superiority of ensemble methods over individual classifiers holds when evaluated using an open-source linguistic representation.
5. Provide a rigorous, quantitative comparison against the published results in Table 2 of the original paper.

---

## IV. Datasets

Following Ahmad et al., our replication uses three distinct primary news datasets collected from online sources, as well as a master combined dataset.

```
+-------------------------------------------------------------------+
|                        Data Collection                            |
|  DS1: ISOT (N=44,182) | DS2: Kaggle (N=20,408) | DS3: Jruvika (N=3,990)
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               Master Combination & Random Shuffling               |
|                    DS4 = DS1 + DS2 + DS3 (N=68,580)               |
+-------------------------------------------------------------------+
```

### A. DS1: ISOT Fake News Dataset
The ISOT dataset [23] was compiled by the University of Victoria Information Security and Object Technology research group. It comprises legitimate news stories extracted from Reuters.com and fraudulent articles collected from untrusted domains flagged by fact-checking platforms such as PolitiFact. Raw totals comprise 21,417 truthful articles and 23,481 fake articles (44,898 raw total). Following HTML stripping, deduplication, and minimum word count filtering ($\ge 20$ words), DS1 contains **44,182 articles** (22,766 Fake, 21,416 Truthful).

### B. DS2: Kaggle Fake News Dataset
DS2 originates from the Kaggle Fake News competition [24]. This dataset contains full article text, headlines, author attributions, and reliability labels across a broad range of domains. The raw training split contains 20,800 records. Following cleaning and filtering, DS2 contains **20,408 articles** (10,022 Fake, 10,386 Truthful). Note that while raw Kaggle records label unreliable articles as $1$ and reliable articles as $0$, our pipeline maps labels to standard research conventions ($0 = \text{Fake}$, $1 = \text{Truthful}$).

### C. DS3: Kaggle Fake News Detection (Jruvika)
DS3 is a smaller, multi-domain benchmark published on Kaggle by Jruvika [25]. The raw dataset contains 4,009 articles covering sports, technology, entertainment, and politics. Following text cleaning and length filtering, DS3 contains **3,990 articles** (2,122 Fake, 1,868 Truthful).

### D. DS4: Combined Master Corpus
To evaluate model generalization across heterogeneous journalistic domains, Ahmad et al. concatenated the three primary datasets into a master dataset. Our pipeline concatenates cleaned DS1, DS2, and DS3 records, applies a global random shuffle with a fixed seed ($\text{random\_state}=42$), and resets indices. DS4 contains **68,580 articles** (34,910 Fake, 33,670 Truthful).

TABLE I summarizes the corpus statistics across all four datasets.

**TABLE I: Summary of Cleaned Dataset Characteristics**
| Dataset | Source / Description | Fake ($y=0$) | Truthful ($y=1$) | Total Articles | Fake Ratio (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DS1** | ISOT Research Dataset [23] | 22,766 | 21,416 | 44,182 | 51.53% |
| **DS2** | Kaggle Fake News Competition [24] | 10,022 | 10,386 | 20,408 | 49.11% |
| **DS3** | Kaggle Fake News Detection [25] | 2,122 | 1,868 | 3,990 | 53.18% |
| **DS4** | Combined Master Corpus (DS1+DS2+DS3) | 34,910 | 33,670 | 68,580 | 50.90% |

---

## V. Methodology

The complete replication pipeline consists of text preprocessing, 93-feature linguistic approximation extraction, stratified splitting, MinMaxScaler normalization, model training with 10-fold cross-validated grid search, and out-of-sample evaluation.

```
+---------------------------------------------------------------+
|                       Raw Text Article                        |
+---------------------------------------------------------------+
                               |
                               v
+---------------------------------------------------------------+
|                      Text Preprocessing                       |
|   HTML unescape, tag removal, whitespace strip, >=20 words   |
+---------------------------------------------------------------+
                               |
                               v
+---------------------------------------------------------------+
|           93-Feature Linguistic Extraction (LIWC-Approx)      |
|  Summary stats, pronouns, affect, cognition, grammar, syntax  |
+---------------------------------------------------------------+
                               |
                               v
+---------------------------------------------------------------+
|           70/30 Stratified Train/Test Split (seed=42)         |
+---------------------------------------------------------------+
            |                                       |
     Train Split (70%)                       Test Split (30%)
            |                                       |
            v                                       v
+-----------------------+              +------------------------+
| Fit & Transform       |              | Transform Only         |
| MinMaxScaler [0, 1]   |              | (Using Fitted Scaler)  |
+-----------------------+              +------------------------+
            |                                       |
            v                                       |
+-----------------------+                           |
| 10-Fold GridSearchCV  |                           |
| Tuning on X_train     |                           |
+-----------------------+                           |
            |                                       |
            v                                       |
+---------------------------------------------------------------+
|               Evaluation on Isolated Test Split               |
|            Accuracy, Precision, Recall, F1-Score              |
+---------------------------------------------------------------+
```

### A. Data Preprocessing
Text cleaning is standardized across all corpora via `src/data_preprocessing.py`:
1. **Entity and Tag Decoding**: HTML entities (e.g., `&amp;`, `&quot;`) are converted to standard ASCII characters using `html.unescape`. HTML tags (`<p>`, `<br>`, `<div>`) are removed via regular expressions.
2. **Metadata Filtering**: Author names, publication dates, publication source tags, and URL strings are discarded to prevent algorithms from memorizing non-textual identifiers.
3. **Headline and Body Fusion**: The headline/title is concatenated with the article body separated by a single space.
4. **Length Pruning**: Following Ahmad et al., articles with empty body text or fewer than 20 total words are excluded.

### B. 93-Feature Open-Source LIWC-Style Approximation
Ahmad et al. [1] employed the commercial LIWC2015 tool [11] to generate 93 discrete and continuous variables. Because LIWC2015 is closed-source and its dictionary is proprietary, our project engineered an open-source approximation in `src/liwc_features.py`. The extractor parses raw text and extracts 87 active linguistic measurements across 10 functional categories, supplemented by 6 constant-zero padding features to construct a 93-dimensional feature vector aligning structurally with LIWC2015's column schema:
- **Text Statistics (6 features)**: Word count ($wc$), words per sentence ($wps$), percentage of words longer than six letters ($sixltr$), total dictionary matches proxy ($dic$), numeral count ($numerals$), and average word length ($avg\_word\_len$).
- **Punctuation Counts (6 features)**: Frequencies of periods ($period$), commas ($comma$), colons ($colon$), exclamation marks ($excl$), question marks ($qmark$), and dashes ($dash$).
- **Function Words & Pronouns (11 features)**: Total pronouns ($pronoun$), personal pronouns ($ppron$), first-person singular ($i$), first-person plural ($we$), second-person ($you$), third-person singular ($shehe$), third-person plural ($they$), impersonal pronouns ($ipron$), articles ($article$), prepositions ($prep$), and auxiliary verbs ($auxverb$).
- **Social and Affective Processes (11 features)**: Social words ($social$, $family$, $friend$, $female$, $male$), general affect ($affect$), positive emotion ($posemo$), negative emotion ($negemo$), anxiety ($anx$), anger, and sadness ($sad$).
- **Cognitive Processes (5 features)**: General cognitive processes ($cogproc$), insight ($insight$), causation ($cause$), discrepancies ($discrep$), and tentative language ($tentat$).
- **Perceptual and Biological Terms (10 features)**: Perceptual verbs ($percept$, $see$, $hear$, $feel$), biological terms ($bio$, $body$, $health$, $sexual$, $ingest$), and basic psychological drives ($drives$).
- **Time and Space Orientation (7 features)**: Past tense markers ($focuspast$), present tense markers ($focuspresent$), future tense markers ($focusfuture$), spatial/motion relativity ($relativ$, $motion$, $space$), and general time references ($time$).
- **Personal Concerns (7 features)**: Work ($work$), achievement ($achieve$), leisure ($leisure$), home ($home$), financial terms ($money$), religion ($relig$), and death ($death$).
- **Grammatical Markers (8 features)**: Verbs ($verb$), adjectives ($adj$), comparatives ($compare$), interrogatives ($interrog$), number words ($number\_wrd$), quantifiers ($quant$), conjunctions ($conjunctions$), and negations ($negate$).
- **Informal Language & Readability (16 features)**: Informal speech ($informal$, $swear$, $netspeak$, $assent$, $nonflu$, $filler$), readability and lexical diversity metrics ($unique\_words\_ratio$, $sentence\_count$, $avg\_sentence\_len$, $stopword\_ratio$), sentiment polarity lexicon scores ($sentiment\_pos$, $sentiment\_neg$, $sentiment\_compound$), URL presence flag ($has\_url$), capitalization ratio ($allcaps\_ratio$), and title-case ratio ($title\_case\_ratio$).
- **Dimensionality Alignment Padding (6 features)**: Fixed constant-zero alignment features ($pad\_88$ through $pad\_93$) explicitly defined in `src/liwc_features.py` to maintain exact 93-feature dimensionality matching the 93-column specification of LIWC2015.
- **Deterministic Order & Dimensionality**: The extractor outputs exactly 93 features (87 active linguistic measurements + 6 constant-zero padding features = 93) with deterministic ordering across all records.

### C. Feature Scaling and Train-Test Split
Each dataset is partitioned into 70% training and 30% testing subsets using stratified sampling with `random_state=42`. Stratification guarantees that the ratio of fake to truthful articles remains identical in both partitions.

To normalize continuous counts and percentage ratios into a uniform $[0, 1]$ interval, scikit-learn's `MinMaxScaler` is employed:
$$x_{\text{scaled}} = \frac{x - x_{\min,\text{train}}}{x_{\max,\text{train}} - x_{\min,\text{train}}}$$
**Held-Out Test Set Isolation and Scaling Protocol**: Scikit-learn's `MinMaxScaler` is fitted across the complete 70% training partition $X_{\text{train}}$. The held-out test partition $X_{\text{test}}$ is transformed strictly using the fitted training extrema ($x_{\min,\text{train}}, x_{\max,\text{train}}$) without parameter re-estimation, ensuring full isolation of the evaluation split. However, because scaling precedes cross-validation, the scaling bounds reflect the entire training split rather than being re-estimated inside each cross-validation fold, as discussed in Section X-B.

### D. Machine Learning Classifiers
We implement the four individual benchmark algorithms specified in Section 2.2 of Ahmad et al.:

1. **Logistic Regression (LR)**: Models class posterior probability via the logistic sigmoid function:
   $$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
   Optimized via the L-BFGS solver with $\ell_2$ regularization penalty controlled by hyperparameter $C$.

2. **Linear Support Vector Machine (LinearSVC)**: Seeks a separating hyperplane minimizing hinge loss with $\ell_2$ regularization:
   $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$
   We use the `LinearSVC` implementation with `dual="auto"` and `max_iter=2000`.

3. **Multilayer Perceptron (MLP)**: A feedforward artificial neural network consisting of an input layer (93 units), one hidden layer with ReLU activation ($h = \max(0, \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$), and a logistic output unit trained via the Adam optimizer with early stopping.

4. **K-Nearest Neighbors (KNN)**: An instance-based non-parametric classifier assigning the label based on the majority class among the $k$ closest training points in feature space according to a distance metric $d(\mathbf{x}, \mathbf{z})$.

### E. Ensemble Methods
We implement the six ensemble classifiers detailed in Section 2.3 of the original paper:

1. **Random Forest (RF)**: An ensemble of $B=100$ decorrelated decision trees constructed using bootstrap samples of the training set. At each split, a random subset of features $m \le \sqrt{p}$ is considered using the Gini impurity criterion:
   $$\text{Gini}(t) = 1 - \sum_{k=0}^{1} p(k|t)^2$$

2. **Bagging Classifier (Bootstrap Aggregating)**: An ensemble of 100 unpruned Decision Trees trained on random sub-samples (sampling ratio 0.8) drawn with replacement from $X_{\text{train}}$. Predictions represent unweighted average votes across all base estimators.

3. **AdaBoost (Adaptive Boosting)**: A sequential boosting algorithm fitting a series of shallow decision tree stumps ($d=1$). At each iteration $m$, misclassified instances are assigned higher sample weights:
   $$w_{i}^{(m+1)} = w_{i}^{(m)} \exp\left(\alpha_m \mathbb{I}(y_i \ne h_m(\mathbf{x}_i))\right)$$
   Final predictions represent a weighted majority vote: $H(\mathbf{x}) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m h_m(\mathbf{x})\right)$.

4. **XGBoost (Extreme Gradient Boosting)**: A scalable tree boosting system optimizing an objective function comprising a convex loss function and tree complexity regularizer $\Omega(f_m) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$. XGBoost uses second-order Taylor expansions to optimize split finding.

5. **Voting Classifier 1 (VC1 - RF + LR + KNN)**: A consensus ensemble combining the tuned predictions of Random Forest, Logistic Regression, and KNN via hard majority voting:
   $$\hat{y}_{\text{VC1}} = \mathbb{I}\left( \hat{y}_{\text{RF}} + \hat{y}_{\text{LR}} + \hat{y}_{\text{KNN}} \ge 2 \right)$$

6. **Voting Classifier 2 (VC2 - LR + LSVM + DT)**: A consensus ensemble combining the predictions of Logistic Regression, Linear SVM, and a CART Decision Tree via hard majority voting:
   $$\hat{y}_{\text{VC2}} = \mathbb{I}\left( \hat{y}_{\text{LR}} + \hat{y}_{\text{LSVM}} + \hat{y}_{\text{DT}} \ge 2 \right)$$

### F. Hyperparameter Optimization via GridSearchCV
Hyperparameters were tuned via scikit-learn's `GridSearchCV` using **10-fold cross-validation** ($k=10$) fitted exclusively on $X_{\text{train}}$. Scoring was guided by cross-validated accuracy. While exhaustive grid search systematically explores discrete hyperparameter configurations, it incurs substantial computational expense compared to randomized search strategies [26]. 

TABLE II specifies the parameter search grids evaluated across all models.

**TABLE II: Hyperparameter Search Grids**
| Model Key | Algorithm | Evaluated Hyperparameter Grid | Combinations | CV Folds | Total CV Fits |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **LR** | Logistic Regression | $C \in \{0.1, 1.0, 10.0\}$, `solver=['lbfgs']`, `max_iter=[1000]` | 3 | 10 | 30 |
| **LSVM** | Linear SVM | $C \in \{0.1, 1.0, 10.0\}$, `max_iter=[2000]` | 3 | 10 | 30 |
| **MLP** | Multilayer Perceptron | $\text{hidden\_layer\_sizes} \in \{(50,), (100,)\}$, $\text{early\_stopping}=[\text{True}]$ | 2 | 10 | 20 |
| **KNN** | K-Nearest Neighbors | $k \in \{3, 5, 7\}$, $\text{metric} \in \{\text{'euclidean'}, \text{'cosine'}\}$ | 6 | 10 | 60 |
| **RF** | Random Forest | $\text{max\_features} \in \{\text{'sqrt'}, \text{'log2'}\}$, $n_{\text{est}}=100$, $\text{criterion}=\text{'gini'}$ | 2 | 10 | 20 |
| **AdaBoost** | AdaBoost | $n_{\text{est}} \in \{50, 100\}$, $\text{learning\_rate} \in \{0.5, 1.0\}$ | 4 | 10 | 40 |
| **XGB** | XGBoost | $\text{max\_depth} \in \{4, 6\}$, $\text{learning\_rate}=[0.1]$, $n_{\text{est}}=[100]$ | 2 | 10 | 20 |
| **Bagging** | Bagging (DTs) | Excluded from GridSearch (Single baseline configuration: $n_{\text{est}}=100$) | — | — | Direct fit (outside CV) |
| **VC1, VC2**| Voting Ensembles | Assembled directly from predictions of tuned base estimators | — | — | Consensus vote (0 fits) |
| **GridSearchCV Total** | | Seven tuned base models | **22** | **10** | **220 CV fits / dataset** |

Across all four datasets, the GridSearchCV hyperparameter search comprised 22 parameter combinations × 10 folds × 4 datasets = **880 cross-validation fits** (alongside 4 direct single-model fits for Bagging and direct majority voting aggregation for VC1 and VC2).

---

## VI. Experimental Setup

The experimental environment was executed under Python 3.14 on a 64-bit Windows workstation equipped with an Intel multi-core processor (12 logical cores) and 16 GB physical RAM. Dependencies included `scikit-learn` (v1.9.1), `xgboost` (v3.4.1), `pandas` (v3.0.6), `numpy` (v2.5.3), and `joblib` (v1.6.0).

To prevent memory exhaustion during 10-fold cross-validation on large datasets (DS1: 30,927 train rows; DS4: 48,006 train rows), cross-validation worker concurrency was controlled using `GRID_SEARCH_N_JOBS = 2` with `pre_dispatch = "n_jobs"`. Inner estimators were constrained to single-threaded execution (`n_jobs=1`), eliminating nested thread thrashing. Garbage collection (`gc.collect()`) was enforced after every model evaluation.

### Performance Metrics
Model performance was evaluated on the test set using standard binary classification metrics:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
where $TP$, $TN$, $FP$, and $FN$ represent true positives, true negatives, false positives, and false negatives, respectively.

---

## VII. Results

### A. Comprehensive Model Evaluation Across Datasets
TABLE III presents the complete experimental results for all ten models across DS1, DS2, DS3, and DS4 following 10-fold cross-validated grid search tuning. A cross-dataset graphical comparison across all ten models is illustrated in Fig. 1 (`results/plots/accuracy_across_datasets.png`).

**TABLE III: Performance Metrics Across All Datasets (GridSearch Tuned)**
| Dataset | Model Key | Model Name | Category | Accuracy | Precision | Recall | F1-Score | Train Time (s) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DS1** | LR | Logistic Regression | Individual | 0.9432 | 0.9283 | 0.9567 | 0.9423 | 4.71s |
| | LSVM | Linear SVM | Individual | 0.9488 | 0.9340 | 0.9625 | 0.9480 | 8.01s |
| | MLP | Multilayer Perceptron | Individual | 0.9649 | 0.9633 | 0.9644 | 0.9638 | 100.92s |
| | KNN | K-Nearest Neighbors | Individual | 0.8939 | 0.8419 | 0.9617 | 0.8978 | 27.63s |
| | RF | Random Forest | Ensemble | 0.9691 | **0.9779** | 0.9580 | 0.9678 | 49.42s |
| | VC1 | Voting Classifier 1 (RF+LR+KNN) | Ensemble | 0.9562 | 0.9389 | **0.9729** | 0.9556 | 4.92s |
| | VC2 | Voting Classifier 2 (LR+LSVM+DT)| Ensemble | 0.9514 | 0.9380 | 0.9634 | 0.9506 | 0.01s |
| | Bagging | Bagging Classifier (DTs) | Ensemble | 0.9599 | 0.9685 | 0.9480 | 0.9582 | 26.83s |
| | AdaBoost| AdaBoost Classifier | Ensemble | 0.9437 | 0.9334 | 0.9518 | 0.9425 | 124.80s |
| | XGB | XGBoost Classifier | Ensemble | **0.9694** | 0.9687 | 0.9681 | **0.9684** | 11.85s |
| **DS2** | LR | Logistic Regression | Individual | 0.8426 | 0.8650 | 0.8184 | 0.8410 | 1.62s |
| | LSVM | Linear SVM | Individual | 0.8546 | 0.8786 | 0.8289 | 0.8530 | 3.92s |
| | MLP | Multilayer Perceptron | Individual | 0.8685 | 0.8850 | 0.8524 | 0.8684 | 27.90s |
| | KNN | K-Nearest Neighbors | Individual | 0.8052 | 0.7837 | 0.8524 | 0.8166 | 6.92s |
| | RF | Random Forest | Ensemble | 0.8805 | **0.9024** | 0.8578 | 0.8796 | 25.66s |
| | VC1 | Voting Classifier 1 (RF+LR+KNN) | Ensemble | 0.8726 | 0.8909 | 0.8543 | 0.8722 | 1.09s |
| | VC2 | Voting Classifier 2 (LR+LSVM+DT)| Ensemble | 0.8520 | 0.8756 | 0.8267 | 0.8504 | 0.00s |
| | Bagging | Bagging Classifier (DTs) | Ensemble | 0.8759 | 0.8916 | 0.8607 | 0.8759 | 12.28s |
| | AdaBoost| AdaBoost Classifier | Ensemble | 0.8282 | 0.8482 | 0.8068 | 0.8270 | 59.97s |
| | XGB | XGBoost Classifier | Ensemble | **0.8912** | 0.8964 | **0.8890** | **0.8927** | 7.85s |
| **DS3** | LR | Logistic Regression | Individual | 0.9265 | 0.8986 | 0.9500 | 0.9236 | 0.36s |
| | LSVM | Linear SVM | Individual | 0.9290 | 0.9060 | 0.9464 | 0.9258 | 0.38s |
| | MLP | Multilayer Perceptron | Individual | 0.9156 | 0.8781 | 0.9518 | 0.9135 | 4.09s |
| | KNN | K-Nearest Neighbors | Individual | 0.9332 | 0.9152 | 0.9446 | 0.9297 | 0.59s |
| | RF | Random Forest | Ensemble | 0.9541 | 0.9407 | 0.9625 | 0.9515 | 4.63s |
| | VC1 | Voting Classifier 1 (RF+LR+KNN) | Ensemble | 0.9440 | 0.9199 | **0.9643** | 0.9416 | 0.07s |
| | VC2 | Voting Classifier 2 (LR+LSVM+DT)| Ensemble | 0.9307 | 0.9049 | 0.9518 | 0.9278 | 0.00s |
| | Bagging | Bagging Classifier (DTs) | Ensemble | 0.9507 | 0.9341 | 0.9625 | 0.9481 | 1.74s |
| | AdaBoost| AdaBoost Classifier | Ensemble | 0.9323 | 0.9194 | 0.9375 | 0.9284 | 13.82s |
| | XGB | XGBoost Classifier | Ensemble | **0.9624** | **0.9558** | **0.9643** | **0.9600** | 4.13s |
| **DS4** | LR | Logistic Regression | Individual | 0.8540 | 0.8412 | 0.8662 | 0.8535 | 7.93s |
| | LSVM | Linear SVM | Individual | 0.8616 | 0.8472 | 0.8761 | 0.8614 | 19.54s |
| | MLP | Multilayer Perceptron | Individual | 0.8999 | 0.8890 | **0.9098** | 0.8993 | 248.35s |
| | KNN | K-Nearest Neighbors | Individual | 0.8288 | 0.7820 | 0.9030 | 0.8381 | 72.76s |
| | RF | Random Forest | Ensemble | **0.9126** | **0.9325** | 0.8861 | **0.9087** | 101.97s |
| | VC1 | Voting Classifier 1 (RF+LR+KNN) | Ensemble | 0.8907 | 0.8768 | 0.9046 | 0.8905 | 12.27s |
| | VC2 | Voting Classifier 2 (LR+LSVM+DT)| Ensemble | 0.8647 | 0.8518 | 0.8771 | 0.8643 | 0.01s |
| | Bagging | Bagging Classifier (DTs) | Ensemble | 0.9066 | 0.9261 | 0.8799 | 0.9024 | 54.49s |
| | AdaBoost| AdaBoost Classifier | Ensemble | 0.8593 | 0.8433 | 0.8762 | 0.8594 | 223.80s |
| | XGB | XGBoost Classifier | Ensemble | 0.9077 | 0.9123 | 0.8984 | 0.9053 | 17.10s |

### B. Average Model Performance Across All Corpora
TABLE IV ranks all ten models by mean accuracy across the four datasets. Tree-based ensembles (XGBoost, Random Forest, Bagging) and Voting Classifier 1 occupy the top four ranks, demonstrating that non-linear feature interaction modeling and variance reduction provide strong decision boundaries over these linguistic features. However, ensemble performance is not uniformly superior across all architectures: AdaBoost (rank 9, mean accuracy 0.8909) underperforms individual baselines including Multilayer Perceptron (0.9122), Linear SVM (0.8985), and Logistic Regression (0.8916). This indicates that shallow boosting stumps ($d=1$) struggle to capture feature interactions as effectively as deeper tree ensembles or continuous neural representations. The relative ranking distribution is illustrated in Fig. 2 (`results/plots/average_accuracy_by_model.png`).

**TABLE IV: Overall Ranked Performance Averaged Across DS1–DS4**
| Rank | Model Key | Model Name | Category | Mean Accuracy | Mean Precision | Mean Recall | Mean F1-Score |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **XGB** | Boosting Classifier (XGBoost) | Ensemble | **0.9327** | 0.9333 | 0.9300 | **0.9316** |
| **2** | **RF** | Random Forest (RF) | Ensemble | **0.9291** | **0.9384** | 0.9161 | **0.9269** |
| **3** | **Bagging** | Bagging Classifier (DTs) | Ensemble | **0.9233** | 0.9301 | 0.9128 | **0.9212** |
| **4** | **VC1** | Voting Classifier 1 (RF+LR+KNN) | Ensemble | **0.9159** | 0.9066 | **0.9240** | **0.9150** |
| **5** | **MLP** | Multilayer Perceptron (MLP) | Individual | **0.9122** | 0.9038 | 0.9196 | **0.9112** |
| **6** | **VC2** | Voting Classifier 2 (LR+LSVM+DT) | Ensemble | **0.8997** | 0.8926 | 0.9048 | **0.8983** |
| **7** | **LSVM** | Linear SVM (LSVM) | Individual | **0.8985** | 0.8915 | 0.9035 | **0.8970** |
| **8** | **LR** | Logistic Regression (LR) | Individual | **0.8916** | 0.8833 | 0.8978 | **0.8901** |
| **9** | **AdaBoost** | Boosting Classifier (AdaBoost) | Ensemble | **0.8909** | 0.8861 | 0.8931 | **0.8893** |
| **10** | **KNN** | K-Nearest Neighbors (KNN) | Individual | **0.8653** | 0.8307 | 0.9154 | **0.8706** |

### C. Baseline vs. GridSearch-Tuned Performance
To quantify the benefit of hyperparameter tuning, TABLE V details the accuracy shifts between baseline (default scikit-learn parameters) and tuned models. A graphical comparison of baseline versus tuned performance is depicted in Fig. 3 (`results/plots/baseline_vs_tuned_accuracy.png`).

**TABLE V: Baseline vs. Tuned Accuracy Gains ($\Delta = \text{Tuned} - \text{Baseline}$)**
| Model Key | DS1 Gain | DS2 Gain | DS3 Gain | DS4 Gain | Cross-Dataset Mean Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LR** | +0.0129 (+1.29%) | **+0.0289 (+2.89%)** | +0.0092 (+0.92%) | +0.0147 (+1.47%) | **+0.0164 (+1.64%)** |
| **LSVM** | +0.0061 (+0.61%) | **+0.0130 (+1.30%)** | +0.0042 (+0.42%) | +0.0082 (+0.82%) | **+0.0079 (+0.79%)** |
| **AdaBoost** | +0.0090 (+0.90%) | **+0.0186 (+1.86%)** | +0.0067 (+0.67%) | +0.0094 (+0.94%) | **+0.0109 (+1.09%)** |
| **VC1** | +0.0089 (+0.89%) | **+0.0149 (+1.49%)** | +0.0117 (+1.17%) | +0.0062 (+0.62%) | **+0.0104 (+1.04%)** |
| **VC2** | +0.0055 (+0.55%) | **+0.0186 (+1.86%)** | +0.0034 (+0.34%) | +0.0094 (+0.94%) | **+0.0092 (+0.92%)** |
| **KNN** | +0.0033 (+0.33%) | +0.0072 (+0.72%) | +0.0034 (+0.34%) | +0.0036 (+0.36%) | **+0.0044 (+0.44%)** |
| **RF** | 0.0000 | 0.0000 | 0.0000 | +0.0019 (+0.19%) | **+0.0005 (+0.05%)** |
| **MLP, XGB** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |

Logistic Regression gained up to +2.89% on DS2 as GridSearch selected $C=10.0$, relaxing the overly restrictive default $\ell_2$ penalty. For KNN, tuning selected the cosine metric over Euclidean distance across all datasets. Tuning the base learners produced compound gains in Voting Classifiers 1 and 2 without necessitating direct ensemble optimization.

---

## VIII. Comparison With the Original Paper

TABLE VI presents a direct numerical comparison between the accuracy reported in Table 2 of Ahmad et al. (2020) [1] and our experimental results. A comparative visualization contrasting our replication results against the figures reported in Ahmad et al. is presented in Fig. 4 (`results/plots/our_results_vs_paper.png`).

**TABLE VI: Direct Comparison of Accuracy: Ahmad et al. (2020) vs. Our Tuned Results**
| Dataset | Model Key | Model Name | Paper Reported (LIWC2015) | Our Replication (LIWC-Approx) | Difference ($\Delta$) | Alignment Assessment |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **DS1** | LR | Logistic Regression | 0.9700 | 0.9432 | -0.0268 | Close Match ($\Delta < 2.7\%$) |
| | LSVM | Linear SVM | 0.9800 | 0.9488 | -0.0312 | Close Match ($\Delta \approx 3.1\%$) |
| | MLP | Multilayer Perceptron | 0.9800 | 0.9649 | -0.0151 | Close Match ($\Delta < 1.5\%$) |
| | KNN | K-Nearest Neighbors | 0.8800 | 0.8939 | +0.0139 | Outperformed (+1.4%) |
| | RF | Random Forest | 0.9900 | 0.9691 | -0.0209 | Close Match ($\Delta \approx 2.1\%$) |
| | VC1 | Voting Classifier 1 | 0.9700 | 0.9562 | -0.0138 | Close Match ($\Delta < 1.4\%$) |
| | VC2 | Voting Classifier 2 | 0.9600 | 0.9514 | -0.0086 | Close Match ($\Delta < 0.9\%$) |
| | Bagging | Bagging Classifier | 0.9800 | 0.9599 | -0.0201 | Close Match ($\Delta \approx 2.0\%$) |
| | AdaBoost| AdaBoost Classifier | 0.9800 | 0.9437 | -0.0363 | Close Match ($\Delta \approx 3.6\%$) |
| | XGB | XGBoost Classifier | 0.9800 | 0.9694 | -0.0106 | Close Match ($\Delta \approx 1.1\%$) |
| **DS2** | LR | Logistic Regression | 0.9100 | 0.8426 | -0.0674 | Moderate ($\Delta \approx 6.7\%$) |
| | LSVM | Linear SVM | 0.3700 | 0.8546 | **+0.4846** | Substantial Divergence (See §IX.B) |
| | MLP | Multilayer Perceptron | 0.3500 | 0.8685 | **+0.5185** | Substantial Divergence (See §IX.B) |
| | KNN | K-Nearest Neighbors | 0.2800 | 0.8052 | **+0.5252** | Substantial Divergence (See §IX.B) |
| | RF | Random Forest | 0.3500 | 0.8805 | **+0.5305** | Substantial Divergence (See §IX.B) |
| | VC1 | Voting Classifier 1 | 0.8800 | 0.8726 | -0.0074 | **Near-Identical** ($\Delta < 0.8\%$) |
| | VC2 | Voting Classifier 2 | 0.8600 | 0.8520 | -0.0080 | **Near-Identical** ($\Delta < 0.8\%$) |
| | Bagging | Bagging Classifier | 0.9400 | 0.8759 | -0.0641 | Moderate ($\Delta \approx 6.4\%$) |
| | AdaBoost| AdaBoost Classifier | 0.9200 | 0.8282 | -0.0918 | Moderate ($\Delta \approx 9.2\%$) |
| | XGB | XGBoost Classifier | 0.9400 | 0.8912 | -0.0488 | Close Match ($\Delta < 4.9\%$) |
| **DS3** | LR | Logistic Regression | 0.9100 | 0.9265 | +0.0165 | Outperformed (+1.7%) |
| | LSVM | Linear SVM | 0.5300 | 0.9290 | **+0.3990** | Substantial Divergence (See §IX.B) |
| | MLP | Multilayer Perceptron | 0.9400 | 0.9156 | -0.0244 | Close Match ($\Delta < 2.5\%$) |
| | KNN | K-Nearest Neighbors | 0.8200 | 0.9332 | **+0.1132** | Outperformed (+11.3%) |
| | RF | Random Forest | 0.9500 | 0.9541 | +0.0041 | **Near-Identical** ($\Delta < 0.5\%$) |
| | VC1 | Voting Classifier 1 | 0.9400 | 0.9440 | +0.0040 | **Near-Identical** ($\Delta < 0.5\%$) |
| | VC2 | Voting Classifier 2 | 0.9200 | 0.9307 | +0.0107 | Near-Identical ($\Delta \approx 1.1\%$) |
| | Bagging | Bagging Classifier | 0.9400 | 0.9507 | +0.0107 | Near-Identical ($\Delta \approx 1.1\%$) |
| | AdaBoost| AdaBoost Classifier | 0.9200 | 0.9323 | +0.0123 | Near-Identical ($\Delta \approx 1.2\%$) |
| | XGB | XGBoost Classifier | 0.9400 | 0.9624 | +0.0224 | Outperformed (+2.2%) |
| **DS4** | LR | Logistic Regression | 0.8700 | 0.8540 | -0.0160 | Close Match ($\Delta < 1.6\%$) |
| | LSVM | Linear SVM | 0.8600 | 0.8616 | +0.0016 | **Near-Identical** ($\Delta < 0.2\%$) |
| | MLP | Multilayer Perceptron | 0.9000 | 0.8999 | -0.0001 | **Near-Identical** ($\Delta < 0.01\%$) |
| | KNN | K-Nearest Neighbors | 0.7700 | 0.8288 | +0.0588 | Outperformed (+5.9%) |
| | RF | Random Forest | 0.9100 | 0.9126 | +0.0026 | **Near-Identical** ($\Delta < 0.3\%$) |
| | VC1 | Voting Classifier 1 | 0.8800 | 0.8907 | +0.0107 | Near-Identical ($\Delta \approx 1.1\%$) |
| | VC2 | Voting Classifier 2 | 0.8500 | 0.8647 | +0.0147 | Near-Identical ($\Delta \approx 1.5\%$) |
| | Bagging | Bagging Classifier | 0.9000 | 0.9066 | +0.0066 | **Near-Identical** ($\Delta < 0.7\%$) |
| | AdaBoost| AdaBoost Classifier | 0.8600 | 0.8593 | -0.0007 | **Near-Identical** ($\Delta < 0.1\%$) |
| | XGB | XGBoost Classifier | 0.8900 | 0.9077 | +0.0177 | Near-Identical ($\Delta \approx 1.8\%$) |

---

## IX. Discussion

The empirical outcomes of our replication provide several insights regarding algorithm performance, linguistic representation, and cross-dataset generalization.

```
       Average Accuracy Comparison (Paper vs. Replication)
=================================================================
  Model      Paper Mean      Our Tuned Mean      Difference
-----------------------------------------------------------------
  XGB          93.75%            93.27%            -0.48%  [Close Match]
  RF           80.00%*           92.91%           +12.91%  [*Paper DS2 drop]
  Bagging      94.00%            92.33%            -1.67%  [Close Match]
  VC1          91.75%            91.59%            -0.16%  [Near-Identical]
  MLP          79.25%*           91.22%           +11.97%  [*Paper DS2 drop]
  VC2          89.75%            89.97%            +0.22%  [Near-Identical]
  LSVM         68.50%*           89.85%           +21.35%  [*Paper DS2/DS3 drop]
  LR           91.50%            89.16%            -2.34%  [Close Match]
  AdaBoost     92.00%            89.09%            -2.91%  [Close Match]
  KNN          68.75%*           86.53%           +17.78%  [*Paper DS2 drop]
=================================================================
* Indicates original paper mean was suppressed by extreme DS2 drop.
```

### A. Generalization Across Heterogeneous Domains (DS4)
The master dataset (DS4) represents the primary stress test for cross-domain fake news detection, merging political, general entertainment, and sports reporting. On DS4, our experimental results match Table 2 of Ahmad et al. across almost every algorithm within an absolute margin of **0.01% to 1.8%**. Multilayer Perceptron achieved 0.8999 (paper: 0.9000), AdaBoost achieved 0.8593 (paper: 0.8600), Linear SVM achieved 0.8616 (paper: 0.8600), Random Forest achieved 0.9126 (paper: 0.9100), and Bagging achieved 0.9066 (paper: 0.9000). 

This level of correspondence is consistent with the hypothesis that ensemble learning architectures provide strong inductive bias when learning decision boundaries over heterogeneous text collections.

### B. Examination of the DS2 Base-Learner Discrepancy
As highlighted in TABLE VI, our results diverge substantially from Ahmad et al. on DS2 for individual base learners:
- The original paper reported anomalous accuracy drops on DS2 for Linear SVM (37%), Multilayer Perceptron (35%), KNN (28%), and Random Forest (35%).
- In our audited pipeline, these same models achieved robust accuracies: Linear SVM (85.46%), MLP (86.85%), KNN (80.52%), and Random Forest (88.05%).
- **Hypothesis and Analysis**: The exact technical cause of the low individual baseline figures reported in Ahmad et al. cannot be definitively determined from the published text or available artifacts. It is notable, however, that raw Kaggle competition data uses an inverted label schema ($1 = \text{Unreliable}, 0 = \text{Reliable}$). If an uncalibrated classification threshold or partial label mismatch affected individual models in the original study, decision tree and voting ensembles might have remained resilient through consensus aggregation. Importantly, on **Voting Classifier 1** (paper: 0.8800 vs. ours: 0.8726) and **Voting Classifier 2** (paper: 0.8600 vs. ours: 0.8520), our results align with the original paper within 0.8%.

### C. Superiority of Gradient Tree Boosting and Precision Dynamics
Across all evaluations, XGBoost demonstrated the highest overall accuracy (mean 93.27%). XGBoost’s capacity to handle feature collinearity, apply $\ell_1/\ell_2$ leaf regularization, and prioritize informative structural features gives it an advantage over single-layer neural networks and linear baselines. 

Regarding precision metrics, Random Forest achieved the highest cross-dataset mean precision (0.9384), followed by XGBoost (0.9333). In our experimental formulation, the positive class is designated as Truthful ($y = 1$), with Fake news designated as the negative class ($y = 0$). Consequently, positive-class precision measures $\text{TP}/(\text{TP} + \text{FP})$, representing the proportion of articles predicted as Truthful that are genuinely authentic. A high positive-class precision indicates high fidelity when verifying legitimate journalism—meaning that Random Forest rarely misclassifies fake articles as truthful (low rate of deceptive stories masquerading as verified content). Importantly, this positive-class precision does not directly establish minimal false-positive fake-news flagging (which would correspond to precision on class 0, or false alarms against truthful articles). Rather, it confirms that Random Forest's positive classifications carry low risk of fake-news contamination.

---

## X. Limitations and Threats to Validity

### A. Limitations
To maintain scientific integrity, several methodological limitations must be acknowledged:

1. **Approximation vs. Proprietary LIWC2015**:
   The original research by Ahmad et al. relied on the commercial LIWC2015 software, which matches input tokens against a proprietary, copyrighted dictionary of 6,400+ validated stems. Because LIWC2015 is closed-source, our replication implemented an open-source approximation (`src/liwc_features.py`) utilizing curated word sets and rule-based heuristics. While our feature set captures the same conceptual categories (affect, cognitive operations, social references, grammar, and informality) across 87 extracted dimensions plus 6 constant-zero padding features, the exact underlying numeric values differ from the proprietary software output. Dimensional parity (93 columns) serves solely as a structural formatting interface and does not confer semantic or numerical equivalence with proprietary LIWC2015 dictionary stems. Our results should therefore be interpreted as validating the general utility of linguistic feature representations rather than an exact numeric reproduction of LIWC2015.

2. **Absence of Sequential Semantics**:
   Linguistic feature representations evaluate frequency statistics and lexical distribution but do not model sequential syntax, contextual semantics, or transformer-based attention representations. While effective for stylistic discrimination, linguistic proxies cannot evaluate factuality against dynamic external knowledge graphs.

### B. Threats to Validity
We systematically identify potential threats to validity:
- **Construct Validity**: Relying on open-source lexicon approximations rather than the proprietary LIWC2015 dictionary introduces potential lexical divergence, although the strong concordance on DS1, DS3, and DS4 supports the validity of the constructed proxy.
- **Internal Validity**: Hyperparameter grids were bounded to discrete search spaces (22 combinations, 880 total CV fits) due to hardware constraints rather than exhaustive continuous tuning. Bagging was held to a fixed 100-estimator configuration. Furthermore, in our feature extraction and training pipeline, scikit-learn's `MinMaxScaler` was fitted across the complete 70% training partition $X_{\text{train}}$ prior to initiating 10-fold cross-validated grid search. Methodologically, the held-out test partition $X_{\text{test}}$ remained strictly isolated throughout (the fitted scaler transformed $X_{\text{test}}$ without re-estimating parameters, preventing any held-out evaluation leakage). However, during internal cross-validation on $X_{\text{train}}$, each validation fold influenced the global $x_{\min}$ and $x_{\max}$ scaling parameters used across folds. The precise empirical impact of this validation-fold leakage on hyperparameter selection was not quantitatively measured in this study because the experiment was not rerun with a fold-isolated `Pipeline`. Consequently, whether fold-specific scaling would alter the selected hyperparameters or downstream test metrics remains an open empirical question for future pipeline evaluations.
- **External Validity**: While DS4 merges three distinct corpora, online disinformation evolves rapidly with changing rhetorical styles and multimodal formats (e.g., memes, video transcripts) that stylometric text analysis alone may not fully capture.
- **Conclusion Validity**: A single fixed 70/30 stratified partition was evaluated per dataset, mirroring the design of Ahmad et al., rather than repeated nested cross-validation across multiple random splits.

---

## XI. Reproducibility and Conclusion

### A. Reproducibility Statement
To ensure scientific transparency, the complete experimental pipeline has been archived. The project preserves: (1) raw and cleaned data preprocessing scripts, (2) the open-source linguistic approximation extractor (generating 87 active linguistic measurements plus 6 constant-zero padding dimensions for 93-dimensional schema alignment), (3) model training and 10-fold cross-validation routines, (4) locked CSV evaluation logs, and (5) automated publication visualization scripts. All random seeds were fixed (`seed=42`) to ensure deterministic reproduction upon re-execution.

### B. Conclusion
This study presented an independent, end-to-end replication and empirical evaluation of the fake news detection methodology proposed by Ahmad et al. (2020). By replacing the closed-source LIWC2015 software with a transparent open-source linguistic approximation (87 active linguistic measurements supplemented by 6 constant-zero padding features for 93-dimensional structural schema alignment), we audited the pipeline across four datasets totaling 68,580 articles.

Our experimental results broadly support the primary findings of the original study:
1. **Tree-Based Ensembles Lead Overall**: Advanced tree-based ensembles (XGBoost, Random Forest, and Bagging) consistently achieved the highest classification accuracies across single-domain and mixed-domain corpora, led by XGBoost (cross-dataset mean accuracy 0.9327) and Random Forest (DS4 accuracy 0.9126). However, ensemble superiority was not universal across all ensemble architectures: AdaBoost (mean accuracy 0.8909) trailed individual classifiers such as Multilayer Perceptron (0.9122), Linear SVM (0.8985), and Logistic Regression (0.8916), demonstrating that shallow boosting stumps do not inherently exceed strong non-linear or regularized individual learners on stylistic linguistic representations.
2. **Master Corpus Generalization**: On the combined DS4 master dataset, our replication aligned with published paper accuracies within 0.01% to 1.8% across nearly all models, supporting the premise that linguistic features provide viable signals across diverse journalistic topics.
3. **Utility of Open Linguistic Proxies**: Competitive classification performance (exceeding 91%–96% on primary benchmarks) can be achieved using accessible, open-source linguistic approximations, providing an open framework for future research.

All source code, feature extractors, cross-validation configurations, evaluation metrics, and analysis scripts have been preserved for open scientific verification.

---

## References

[1] H. Ahmad, M. U. Asghar, M. Z. Asghar, A. Khan, and A. H. Mosavi, "Fake News Detection Using Machine Learning Ensemble Methods," *Complexity*, vol. 2020, Article ID 8885861, 11 pages, 2020.

[2] D. M. J. Lazer, M. A. Baum, Y. Benkler, et al., "The science of fake news," *Science*, vol. 359, no. 6380, pp. 1094–1096, 2018.

[3] S. Vosoughi, D. Roy, and S. Aral, "The spread of true and false news online," *Science*, vol. 359, no. 6380, pp. 1146–1151, 2018.

[4] H. Allcott and M. Gentzkow, "Social media and fake news in the 2016 election," *Journal of Economic Perspectives*, vol. 31, no. 2, pp. 211–236, 2017.

[5] H. Ahmed, I. Traore, and S. Saad, "Detection of online fake news using n-gram analysis and machine learning techniques," in *Proceedings of the International Conference on Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments*, pp. 127–138, Springer, 2017.

[6] K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, "Fake news detection on social media: A data mining perspective," *ACM SIGKDD Explorations Newsletter*, vol. 19, no. 1, pp. 22–36, 2017.

[7] N. K. Conroy, V. L. Rubin, and Y. Chen, "Automatic deception detection: Methods for finding fake news," *Proceedings of the Association for Information Science and Technology*, vol. 52, no. 1, pp. 1–4, 2015.

[8] V. L. Rubin, N. Conroy, Y. Chen, and S. Cornwell, "Fake news or truth? Using satirical cues to detect potentially misleading news," in *Proceedings of the Second Workshop on Computational Approaches to Deception Detection*, pp. 7–17, San Diego, CA, USA, 2016.

[9] P. Bühlmann, "Bagging, boosting and ensemble methods," in *Handbook of Computational Statistics*, pp. 985–1022, Springer, Berlin, Germany, 2012.

[10] D. Ruta and B. Gabrys, "Classifier selection for majority voting," *Information Fusion*, vol. 6, no. 1, pp. 63–81, 2005.

[11] J. W. Pennebaker, R. L. Boyd, K. Jordan, and K. Blackburn, "The development and psychometric properties of LIWC2015," University of Texas at Austin, Austin, TX, USA, Tech. Rep., 2015.

[12] V. Pérez-Rosas, B. Kleinberg, A. Lefevre, and R. Mihalcea, "Automatic detection of fake news," in *Proceedings of the 27th International Conference on Computational Linguistics (COLING)*, pp. 3391–3401, 2018.

[13] T. M. Mitchell, *The Discipline of Machine Learning*, Carnegie Mellon University, Pittsburgh, PA, USA, 2006.

[14] N. Cristianini and J. Shawe-Taylor, *An Introduction to Support Vector Machines and Other Kernel-based Learning Methods*, Cambridge University Press, Cambridge, UK, 2000.

[15] L. Breiman, J. Friedman, R. Olshen, and C. Stone, *Classification and Regression Trees*, Wadsworth, Belmont, CA, 1984.

[16] L. Breiman, "Bagging predictors," *Machine Learning*, vol. 24, no. 2, pp. 123–140, 1996.

[17] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[18] B. Gregorutti, B. Michel, and P. Saint-Pierre, "Correlation and variable importance in random forests," *Statistics and Computing*, vol. 27, no. 3, pp. 659–678, 2017.

[19] R. E. Schapire, "A brief introduction to boosting," in *Proceedings of the 16th International Joint Conference on Artificial Intelligence (IJCAI)*, vol. 99, pp. 1401–1406, 1999.

[20] T. Hastie, S. Rosset, J. Zhu, and H. Zou, "Multi-class AdaBoost," *Statistics and Its Interface*, vol. 2, no. 3, pp. 349–360, 2009.

[21] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794, San Francisco, CA, USA, 2016.

[22] L. Lam and S. Y. Suen, "Application of majority voting to pattern recognition: An analysis of its behavior and performance," *IEEE Transactions on Systems, Man, and Cybernetics - Part A: Systems and Humans*, vol. 27, no. 5, pp. 553–568, 1997.

[23] H. Ahmed, I. Traore, and S. Saad, "Detecting opinion spams and fake news using text classification," *Security and Privacy*, vol. 1, no. 1, p. e9, 2018.

[24] Kaggle, "Fake News: Build a system to identify unreliable news articles," Kaggle, San Francisco, CA, USA, 2018. [Online]. Available: https://www.kaggle.com/c/fake-news

[25] Kaggle, "Fake News Detection Dataset," Kaggle, San Francisco, CA, USA, 2018. [Online]. Available: https://www.kaggle.com/jruvika/fake-news-detection

[26] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *Journal of Machine Learning Research*, vol. 13, pp. 281–305, 2012.
