"""
src/train_models.py
====================
Full training + evaluation pipeline for the Ahmad et al. (2020) replication.

Usage
-----
# Primary replication (LIWC-approximation features):
python -m src.train_models

# Supplementary TF-IDF experiment:
python -m src.train_models --mode tfidf

# Specific datasets only:
python -m src.train_models --datasets DS1 DS3

# Skip grid search (faster, for quick smoke tests):
python -m src.train_models --no-gridsearch

Outputs
-------
results/evaluation_metrics.csv   – per-model per-dataset accuracy/prec/recall/F1
results/best_params.csv          – best hyperparameters from GridSearchCV
results/paper_comparison.csv     – our results vs published paper numbers
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    PARAM_GRIDS,
    RANDOM_STATE,
    RESULTS_DIR,
    CV_FOLDS,
)
from src.feature_pipeline import load_features
from src.models import (
    MODEL_CATALOGUE,
    ORDERED_MODEL_KEYS,
    make_lr,
    make_lsvm,
    make_mlp,
    make_knn,
    make_rf,
    make_bagging,
    make_adaboost,
    make_xgb,
    grid_search,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_NAMES = ["DS1", "DS2", "DS3", "DS4"]

# ---------------------------------------------------------------------------
# Published paper results (Table 2 of Ahmad et al. 2020) — Accuracy only
# These are the ORIGINAL paper numbers; do NOT alter them.
# ---------------------------------------------------------------------------
PAPER_RESULTS_ACCURACY = {
    # Model key: {DS1, DS2, DS3, DS4}
    # Verified against Table 2 of Ahmad et al. (2020), Complexity, Article ID 8885861
    "LR":       {"DS1": 0.97, "DS2": 0.91, "DS3": 0.91, "DS4": 0.87},
    "LSVM":     {"DS1": 0.98, "DS2": 0.37, "DS3": 0.53, "DS4": 0.86},
    "MLP":      {"DS1": 0.98, "DS2": 0.35, "DS3": 0.94, "DS4": 0.90},
    "KNN":      {"DS1": 0.88, "DS2": 0.28, "DS3": 0.82, "DS4": 0.77},
    "RF":       {"DS1": 0.99, "DS2": 0.35, "DS3": 0.95, "DS4": 0.91},
    "VC1":      {"DS1": 0.97, "DS2": 0.88, "DS3": 0.94, "DS4": 0.88},
    "VC2":      {"DS1": 0.96, "DS2": 0.86, "DS3": 0.92, "DS4": 0.85},
    "Bagging":  {"DS1": 0.98, "DS2": 0.94, "DS3": 0.94, "DS4": 0.90},
    "AdaBoost": {"DS1": 0.98, "DS2": 0.92, "DS3": 0.92, "DS4": 0.86},
    "XGB":      {"DS1": 0.98, "DS2": 0.94, "DS3": 0.94, "DS4": 0.89},
}


# ---------------------------------------------------------------------------
# Evaluate a fitted model
# ---------------------------------------------------------------------------

def evaluate(
    dataset_name: str,
    model_key: str,
    display_name: str,
    category: str,
    mode: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    train_time: float,
) -> Dict[str, Any]:
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0, average="binary")
    rec  = recall_score(y_test, y_pred, zero_division=0, average="binary")
    f1   = f1_score(y_test, y_pred, zero_division=0, average="binary")
    cm   = confusion_matrix(y_test, y_pred)

    logger.info(
        f"  [{dataset_name}][{mode}] {display_name:38s} "
        f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}  ({train_time:.1f}s)"
    )
    return {
        "Dataset":     dataset_name,
        "Mode":        mode,
        "Category":    category,
        "Model_Key":   model_key,
        "Model":       display_name,
        "Accuracy":    round(acc, 4),
        "Precision":   round(prec, 4),
        "Recall":      round(rec, 4),
        "F1_Score":    round(f1, 4),
        "Train_Time_s": round(train_time, 2),
    }


# ---------------------------------------------------------------------------
# Training loop for one dataset
# ---------------------------------------------------------------------------

def train_and_evaluate_dataset(
    dataset_name: str,
    mode: str = "liwc",
    use_gridsearch: bool = True,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Train all 10 models on *dataset_name* and return evaluation records.

    When use_gridsearch=True:
      - LR, LSVM, MLP, KNN, RF, AdaBoost, XGB are tuned via GridSearchCV(cv=10).
      - Bagging is intentionally EXCLUDED from GridSearchCV (single meaningful
        combo, prohibitively slow on DS1/DS4).  It is always trained directly
        with its default parameters and its test metrics are still recorded.
      - VC1 (RF+LR+KNN) and VC2 (LR+LSVM+DT) are built from the best tuned
        base learners and evaluated on X_test without any additional fitting.
      - X_test and y_test are NEVER passed to GridSearchCV.

    Returns
    -------
    metrics_records : list of dicts (one per model)
    best_params     : {model_key: param_dict}
    """
    logger.info(f"\n{'='*65}\n  Training on {dataset_name} [{mode.upper()}]\n{'='*65}")

    data     = load_features(dataset_name, mode=mode)
    X_train  = data["X_train"]
    X_test   = data["X_test"]
    y_train  = data["y_train"]
    y_test   = data["y_test"]
    exp_mode = data["mode"]   # "LIWC-Approx" or "TF-IDF"

    records: Dict[str, Dict] = {}  # keyed by model_key to preserve order
    best_params: Dict[str, Dict] = {}
    fitted: Dict[str, Any] = {}  # store fitted models for voting classifiers

    def _fit_eval(key: str, cat: str, name: str, clf,
                  do_gs: bool = False, pg_key: Optional[str] = None):
        """
        Fit *clf*, optionally tuning via GridSearchCV on X_train/y_train only.
        Evaluate on X_test.  Append result to records.
        """
        t0 = time.time()
        if do_gs and pg_key and pg_key in PARAM_GRIDS:
            # GridSearchCV sees ONLY X_train / y_train — no leakage
            clf, params, _ = grid_search(clf, PARAM_GRIDS[pg_key], X_train, y_train)
            best_params[key] = params
        else:
            clf.fit(X_train, y_train)
            best_params[key] = {}
        y_pred = clf.predict(X_test)
        fitted[key] = clf
        records[key] = evaluate(
            dataset_name, key, name, cat, exp_mode, y_test, y_pred, time.time() - t0
        )
        gc.collect()
        return clf

    # -------- Individual learners --------
    _fit_eval("LR",   "Individual", "Logistic Regression (LR)",   make_lr(),   use_gridsearch, "LR")
    _fit_eval("LSVM", "Individual", "Linear SVM (LSVM)",          make_lsvm(), use_gridsearch, "LSVM")
    _fit_eval("MLP",  "Individual", "Multilayer Perceptron (MLP)", make_mlp(),  use_gridsearch, "MLP")
    _fit_eval("KNN",  "Individual", "K-Nearest Neighbors (KNN)",   make_knn(),  use_gridsearch, "KNN")

    # Base DT for Voting 2 (not independently evaluated, not grid-searched)
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    fitted["DT"] = dt

    # -------- Ensemble: Random Forest (GridSearchCV when use_gridsearch=True) --------
    _fit_eval("RF", "Ensemble", "Random Forest (RF)", make_rf(), use_gridsearch, "RF")

    # -------- Voting Classifier 1: best RF + best LR + best KNN (hard majority) --------
    # Uses the already-fitted (and tuned, when use_gridsearch=True) base learners.
    # Predictions are generated on the untouched X_test only.
    t0 = time.time()
    vc1_votes = (
        fitted["RF"].predict(X_test)
        + fitted["LR"].predict(X_test)
        + fitted["KNN"].predict(X_test)
    )
    y_vc1 = (vc1_votes >= 2).astype(int)
    records["VC1"] = evaluate(
        dataset_name, "VC1", "Voting Classifier 1 (RF+LR+KNN)",
        "Ensemble", exp_mode, y_test, y_vc1, time.time() - t0
    )

    # -------- Voting Classifier 2: best LR + best LSVM + DT (hard majority) --------
    t0 = time.time()
    vc2_votes = (
        fitted["LR"].predict(X_test)
        + fitted["LSVM"].predict(X_test)
        + fitted["DT"].predict(X_test)
    )
    y_vc2 = (vc2_votes >= 2).astype(int)
    records["VC2"] = evaluate(
        dataset_name, "VC2", "Voting Classifier 2 (LR+LSVM+DT)",
        "Ensemble", exp_mode, y_test, y_vc2, time.time() - t0
    )

    # Free base learner models as voting classifiers are evaluated
    fitted.clear()
    gc.collect()

    # -------- Bagging: ALWAYS trained directly — never GridSearchCV --------
    # Reason: single meaningful combo, extremely slow on DS1/DS4 with 10-fold CV.
    # Metrics are still recorded and included in the tuned results table.
    logger.info(f"  [{dataset_name}] Bagging: training directly (excluded from GridSearchCV by design)")
    _fit_eval("Bagging",  "Ensemble", "Bagging Classifier (DTs)",
              make_bagging(), do_gs=False)   # do_gs=False regardless of use_gridsearch

    # -------- AdaBoost & XGBoost (GridSearchCV when use_gridsearch=True) --------
    _fit_eval("AdaBoost", "Ensemble", "Boosting Classifier (AdaBoost)", make_adaboost(), use_gridsearch, "AdaBoost")
    _fit_eval("XGB",      "Ensemble", "Boosting Classifier (XGBoost)",  make_xgb(),      use_gridsearch, "XGB")

    # Return in the paper-defined model order
    ordered_keys = [
        "LR", "LSVM", "MLP", "KNN",
        "RF", "VC1", "VC2", "Bagging", "AdaBoost", "XGB",
    ]
    gc.collect()
    return [records[k] for k in ordered_keys if k in records], best_params


# ---------------------------------------------------------------------------
# Paper comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(df_results: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Build a side-by-side comparison of our results vs. published paper results.
    Only accuracy is available from the paper for all models and datasets.
    """
    rows = []
    for key in ORDERED_MODEL_KEYS:
        paper_row = PAPER_RESULTS_ACCURACY.get(key, {})
        for ds in ["DS1", "DS2", "DS3", "DS4"]:
            our_row = df_results[
                (df_results["Dataset"] == ds) & (df_results["Model_Key"] == key)
            ]
            our_acc  = our_row["Accuracy"].values[0] if len(our_row) else None
            paper_acc = paper_row.get(ds)
            diff     = round(our_acc - paper_acc, 4) if (our_acc is not None and paper_acc is not None) else None
            note     = "LIWC-Approximation" if mode == "liwc" else "TF-IDF (Supplementary)"
            rows.append({
                "Dataset":          ds,
                "Model_Key":        key,
                "Model":            df_results[df_results["Model_Key"] == key]["Model"].values[0] if len(df_results[df_results["Model_Key"] == key]) else key,
                "Paper_Accuracy":   paper_acc,
                "Our_Accuracy":     our_acc,
                "Difference":       diff,
                "Replication_Note": note,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Print markdown tables (matching Tables 2–5 in the paper)
# ---------------------------------------------------------------------------

METRIC_TABLE_CONFIGS = [
    ("Table 2 (paper replica): Overall Accuracy", "Accuracy"),
    ("Table 3 (paper replica): Precision",        "Precision"),
    ("Table 4 (paper replica): Recall",           "Recall"),
    ("Table 5 (paper replica): F1-Score",         "F1_Score"),
]

MODEL_ORDER = [
    ("Individual", "LR",       "Logistic Regression (LR)"),
    ("Individual", "LSVM",     "Linear SVM (LSVM)"),
    ("Individual", "MLP",      "Multilayer Perceptron (MLP)"),
    ("Individual", "KNN",      "K-Nearest Neighbors (KNN)"),
    ("Ensemble",   "RF",       "Random Forest (RF)"),
    ("Ensemble",   "VC1",      "Voting Classifier 1 (RF+LR+KNN)"),
    ("Ensemble",   "VC2",      "Voting Classifier 2 (LR+LSVM+DT)"),
    ("Ensemble",   "Bagging",  "Bagging Classifier (DTs)"),
    ("Ensemble",   "AdaBoost", "Boosting Classifier (AdaBoost)"),
    ("Ensemble",   "XGB",      "Boosting Classifier (XGBoost)"),
]


def print_metric_tables(df: pd.DataFrame) -> str:
    """Return markdown-formatted Tables 2–5."""
    lines = []
    for title, col in METRIC_TABLE_CONFIGS:
        lines.append(f"\n### {title}\n")
        header = "| Model | DS1 | DS2 | DS3 | DS4 |"
        sep    = "| :--- | :---: | :---: | :---: | :---: |"
        lines += [header, sep]
        curr_cat = None
        for cat, key, name in MODEL_ORDER:
            if cat != curr_cat:
                lines.append(f"| **{cat} Learners** | | | | |")
                curr_cat = cat
            vals = []
            for ds in ["DS1", "DS2", "DS3", "DS4"]:
                sub = df[(df["Dataset"] == ds) & (df["Model_Key"] == key)]
                v   = sub[col].values[0] if len(sub) else float("nan")
                vals.append("—" if np.isnan(v) else f"{v:.2f}")
            lines.append(f"| {name} | {' | '.join(vals)} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    datasets: List[str] = DATASET_NAMES,
    mode: str = "liwc",
    use_gridsearch: bool = True,
) -> pd.DataFrame:
    """
    Run the full experiment across all requested datasets.

    Output paths depend on the run type:
      - Baseline  → evaluation_metrics.csv, paper_comparison.csv
      - Tuned/GS  → evaluation_metrics_tuned.csv, paper_comparison_tuned.csv,
                     best_params.csv

    This ensures that a tuned run NEVER overwrites the existing baseline CSVs.

    Returns the combined metrics DataFrame.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Choose output filenames ----------
    if use_gridsearch:
        metrics_filename     = "evaluation_metrics_tuned.csv"
        comparison_filename  = "paper_comparison_tuned.csv"
        run_type_label       = "Hyperparameter Search (GridSearchCV)"
    else:
        metrics_filename     = "evaluation_metrics.csv"
        comparison_filename  = "paper_comparison.csv"
        run_type_label       = "Baseline (Default Parameters)"

    all_records: List[Dict] = []
    all_best_params: List[Dict] = []

    for ds in datasets:
        records, best_params = train_and_evaluate_dataset(
            ds, mode=mode, use_gridsearch=use_gridsearch,
        )
        all_records.extend(records)
        for key, params in best_params.items():
            all_best_params.append({"Dataset": ds, "Model_Key": key, **params})
        gc.collect()

    df_metrics = pd.DataFrame(all_records)
    df_params  = pd.DataFrame(all_best_params)

    # ---------- Save results ----------
    metrics_path = RESULTS_DIR / metrics_filename
    df_metrics.to_csv(metrics_path, index=False)
    logger.info(f"\nSaved metrics  → {metrics_path}")

    # Also mirror to data/processed/ for notebook convenience
    processed_results_path = DATA_PROCESSED / metrics_filename
    df_metrics.to_csv(processed_results_path, index=False)
    logger.info(f"  (mirror)     → {processed_results_path}")

    if use_gridsearch:
        params_path = RESULTS_DIR / "best_params.csv"
        df_params.to_csv(params_path, index=False)
        logger.info(f"Saved params   → {params_path}")

    # ---------- Paper comparison ----------
    df_cmp = build_comparison_table(df_metrics, mode)
    cmp_path = RESULTS_DIR / comparison_filename
    df_cmp.to_csv(cmp_path, index=False)
    logger.info(f"Saved comparison → {cmp_path}")

    # ---------- Print tables ----------
    tables = print_metric_tables(df_metrics)
    mode_label = (
        "LIWC-Approximation" if mode == "liwc"
        else "TF-IDF (Supplementary, NOT paper replication)"
    )
    print(f"\n{'='*70}")
    print(f"  RESULTS [{run_type_label}] — Feature mode: {mode_label}")
    print("="*70)
    print(tables)
    print("="*70)

    return df_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate models.")
    p.add_argument("--mode",    choices=["liwc", "tfidf"], default="liwc",
                   help="Feature extraction mode (default: liwc)")
    p.add_argument("--datasets", nargs="+", default=DATASET_NAMES,
                   help="Which datasets to evaluate (default: DS1 DS2 DS3 DS4)")
    p.add_argument("--gridsearch", action="store_true", default=False,
                   help="Enable GridSearchCV with 10-fold CV (default: False, runs baseline)")
    p.add_argument("--no-gridsearch", action="store_true", default=False,
                   help="Explicitly disable GridSearchCV (baseline mode)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    use_gs = args.gridsearch and (not args.no_gridsearch)
    run_experiment(
        datasets=args.datasets,
        mode=args.mode,
        use_gridsearch=use_gs,
    )
