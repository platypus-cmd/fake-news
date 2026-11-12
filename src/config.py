"""
src/config.py
=============
Central configuration constants for the fake-news replication project.
Edit values here rather than scattering magic numbers across modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels up from this file: src/ -> fake_news/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
RESULTS_DIR    = PROJECT_ROOT / "results"
PLOTS_DIR      = RESULTS_DIR / "plots"
NOTEBOOKS_DIR  = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Dataset CSV file names
# ---------------------------------------------------------------------------
PROCESSED_FILES = {
    "DS1": DATA_PROCESSED / "DS1_cleaned.csv",
    "DS2": DATA_PROCESSED / "DS2_cleaned.csv",
    "DS3": DATA_PROCESSED / "DS3_cleaned.csv",
    "DS4": DATA_PROCESSED / "DS4_combined.csv",
}

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE    = 0.30
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Label convention (shared across all modules)
# 0 = Fake / Unreliable
# 1 = Truthful / Real / Reliable
# ---------------------------------------------------------------------------
LABEL_FAKE    = 0
LABEL_TRUTH   = 1
LABEL_NAMES   = {0: "Fake", 1: "Truthful"}

# ---------------------------------------------------------------------------
# LIWC / Linguistic Feature Extraction
# ---------------------------------------------------------------------------
# The paper uses LIWC2015 (commercial tool) for 93 linguistic features.
# Set LIWC_DIC_PATH to the path of a LIWC .dic file if available.
# If None, the pipeline will gracefully fall back and report the limitation.
LIWC_DIC_PATH = None          # e.g. Path("data/raw/LIWC2015.dic")
N_LIWC_FEATURES = 93

# ---------------------------------------------------------------------------
# TF-IDF fallback experiment (NOT the paper replication)
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 5_000
TFIDF_NGRAM_RANGE  = (1, 2)

# ---------------------------------------------------------------------------
# Model hyperparameter grids (small grids suitable for a college project)
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "LR": {
        "C":        [0.1, 1.0, 10.0],
        "solver":   ["lbfgs"],
        "max_iter": [1000],
    },
    "LSVM": {
        "C":        [0.1, 1.0, 10.0],
        "max_iter": [2000],
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,)],
        "max_iter":           [200],
        "early_stopping":     [True],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "metric":      ["euclidean", "cosine"],
    },
    "RF": {
        "n_estimators": [100],
        "criterion":    ["gini"],
        "max_features": ["sqrt", "log2"],   # expanded from ["sqrt"] only
    },
    # Bagging is intentionally excluded from GridSearchCV:
    # it has only one meaningful parameter combination and is very slow on
    # large datasets (DS1/DS4).  It is trained directly with default params.
    "AdaBoost": {
        "n_estimators":  [50, 100],
        "learning_rate": [0.5, 1.0],
    },
    "XGB": {
        "n_estimators":  [100],
        "learning_rate": [0.1],
        "max_depth":     [4, 6],            # expanded from [6] only
    },
}

CV_FOLDS = 10  # 10-fold cross-validation for ensemble learners (paper §2.5)

# ---------------------------------------------------------------------------
# Parallelism & Memory control for GridSearchCV
# ---------------------------------------------------------------------------
# Set to 2 to limit concurrent CV worker processes on laptops with 16 GB RAM.
# This prevents RAM exhaustion (99% usage) and avoids CPU thread thrashing.
GRID_SEARCH_N_JOBS = 2
GRID_SEARCH_PRE_DISPATCH = "n_jobs"  # Prevents pre-allocating excess tasks in memory queue
