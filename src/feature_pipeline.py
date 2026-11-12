"""
src/feature_pipeline.py
========================
Unified feature pipeline for the Ahmad et al. (2020) replication.

IMPORTANT — Two distinct experiment modes:
-------------------------------------------
1. LIWC-Approximation (primary replication path)
   - Extracts 93 open-source linguistic features using src/liwc_features.py.
   - These approximate LIWC2015 categories but use different word lists.
   - Labelled "LIWC-Approx" in all result tables.
   - Results should NOT be presented as exact reproductions of the paper.

2. TF-IDF (optional supplementary experiment)
   - TfidfVectorizer(max_features=5000, ngram_range=(1,2)) as used in the
     previous session's exploratory run.
   - Labelled "TF-IDF" in all result tables – clearly separate from paper.

Outputs (per dataset):
   data/processed/DS{1,2,3,4}_features.csv   – LIWC-approx features + label
   models/                                   – scaler / split info

Usage:
   python -m src.feature_pipeline             # LIWC-approx for all datasets
   python -m src.feature_pipeline --tfidf     # TF-IDF mode (supplementary)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Local imports
from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    RESULTS_DIR,
    PROCESSED_FILES,
    TEST_SIZE,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    LIWC_DIC_PATH,
)
from src.liwc_features import extract_features_dataframe, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_NAMES = ["DS1", "DS2", "DS3", "DS4"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_cleaned(dataset_name: str) -> pd.DataFrame:
    """Load a cleaned CSV from data/processed/."""
    path = PROCESSED_FILES.get(dataset_name)
    if path is None:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Choose from {DATASET_NAMES}.")
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}\n"
            "Run src/data_preprocessing.py first."
        )
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    if "label" not in df.columns:
        raise ValueError(f"'label' column missing in {path}.")
    logger.info(f"  Loaded {dataset_name}: {len(df):,} rows, label dist: {df['label'].value_counts().to_dict()}")
    return df


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """70/30 stratified split. Fits on train only (no leakage)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def fit_and_scale(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler_path: Path,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fit MinMaxScaler on X_train only, then transform both sets.
    Saves the fitted scaler to scaler_path.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Scaler saved → {scaler_path}")
    return X_train_sc, X_test_sc, scaler


def save_split(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    prefix: str = "liwc",
) -> None:
    """
    Save train/test splits as CSVs under data/processed/.
    Columns: feature_names + 'label'.
    """
    ds_key = dataset_name.lower()
    for split_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        out = DATA_PROCESSED / f"{ds_key}_{prefix}_{split_name}.csv"
        df_out = pd.DataFrame(X, columns=feature_names)
        df_out.insert(0, "label", y)
        df_out.to_csv(out, index=False)
        logger.info(f"  Saved {split_name} split → {out}  ({len(df_out):,} rows)")


# ---------------------------------------------------------------------------
# LIWC-Approximation pipeline
# ---------------------------------------------------------------------------

def run_liwc_pipeline(dataset_name: str) -> Dict:
    """
    Extract LIWC-approximation features for one dataset,
    split 70/30 stratified, scale, and save.
    """
    logger.info(f"\n{'='*60}\n[LIWC-Approx] Processing {dataset_name}\n{'='*60}")

    if LIWC_DIC_PATH is not None and Path(LIWC_DIC_PATH).exists():
        logger.warning(
            "LIWC_DIC_PATH is set but exact LIWC2015 parsing is not yet "
            "implemented in this open-source replication. "
            "Falling back to LIWC-approximation features."
        )

    df = load_cleaned(dataset_name)

    # ------ Feature extraction ------
    feat_df = extract_features_dataframe(df, text_col="text", label_col="label")

    # Save full feature CSV (all rows, unscaled, with label)
    feat_path = DATA_PROCESSED / f"{dataset_name}_features.csv"
    feat_df.to_csv(feat_path, index=False)
    logger.info(f"  Full feature CSV saved → {feat_path}")

    X = feat_df[FEATURE_NAMES].values.astype(np.float64)
    y = feat_df["label"].values.astype(int)

    # ------ Split ------
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    logger.info(
        f"  Train: {len(y_train):,} | Test: {len(y_test):,} | "
        f"Train fake%: {(y_train==0).mean()*100:.1f}%"
    )

    # ------ Scale (fit on train only) ------
    scaler_path = MODELS_DIR / f"scaler_liwc_{dataset_name.lower()}.joblib"
    X_train_sc, X_test_sc, scaler = fit_and_scale(X_train, X_test, scaler_path)

    # ------ Save splits ------
    save_split(dataset_name, X_train_sc, X_test_sc, y_train, y_test, FEATURE_NAMES, prefix="liwc")

    return {
        "dataset_name":  dataset_name,
        "X_train":       X_train_sc,
        "X_test":        X_test_sc,
        "y_train":       y_train,
        "y_test":        y_test,
        "scaler":        scaler,
        "feature_names": FEATURE_NAMES,
        "mode":          "LIWC-Approx",
    }


# ---------------------------------------------------------------------------
# TF-IDF pipeline (optional supplementary)
# ---------------------------------------------------------------------------

def run_tfidf_pipeline(dataset_name: str) -> Dict:
    """
    TF-IDF feature extraction (supplementary experiment, NOT the paper method).
    Uses the existing sparse-matrix cache from the previous feature_extraction.py session.
    """
    from scipy.sparse import load_npz, save_npz, csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    logger.info(f"\n{'='*60}\n[TF-IDF Supplementary] Processing {dataset_name}\n{'='*60}")

    ds_key = dataset_name.lower()
    train_npz = MODELS_DIR / f"X_train_{ds_key}.npz"
    test_npz  = MODELS_DIR / f"X_test_{ds_key}.npz"
    ytrain_npy = MODELS_DIR / f"y_train_{ds_key}.npy"
    ytest_npy  = MODELS_DIR / f"y_test_{ds_key}.npy"
    scaler_path = MODELS_DIR / f"scaler_{ds_key}.joblib"
    vec_path    = MODELS_DIR / f"vectorizer_{ds_key}.joblib"

    if all(p.exists() for p in [train_npz, test_npz, ytrain_npy, ytest_npy, vec_path]):
        logger.info("  Loading cached TF-IDF matrices …")
        X_train = load_npz(str(train_npz))
        X_test  = load_npz(str(test_npz))
        y_train = np.load(str(ytrain_npy))
        y_test  = np.load(str(ytest_npy))
        scaler  = joblib.load(str(scaler_path)) if scaler_path.exists() else None
        vectorizer = joblib.load(str(vec_path))
    else:
        df = load_cleaned(dataset_name)
        X_text, _, y_all, _ = train_test_split(
            df["text"], df["label"],
            test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
        )
        # Re-do split properly
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["text"].values, df["label"].values,
            test_size=TEST_SIZE, stratify=df["label"].values, random_state=RANDOM_STATE
        )
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test  = vectorizer.transform(X_test_text)
        joblib.dump(vectorizer, str(vec_path))
        scaler = None

    return {
        "dataset_name":  dataset_name,
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "scaler":        scaler,
        "vectorizer":    vectorizer,
        "feature_names": None,  # sparse, no fixed names
        "mode":          "TF-IDF",
    }


# ---------------------------------------------------------------------------
# Convenience loader used by train_models.py
# ---------------------------------------------------------------------------

def load_features(dataset_name: str, mode: str = "liwc") -> Dict:
    """
    Load pre-extracted features for a dataset.

    Parameters
    ----------
    dataset_name : str  – "DS1" | "DS2" | "DS3" | "DS4"
    mode         : str  – "liwc" | "tfidf"

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, scaler, mode
    """
    ds_key = dataset_name.lower()
    if mode == "liwc":
        train_path = DATA_PROCESSED / f"{ds_key}_liwc_train.csv"
        test_path  = DATA_PROCESSED / f"{ds_key}_liwc_test.csv"
        scaler_path = MODELS_DIR / f"scaler_liwc_{ds_key}.joblib"
        if not train_path.exists():
            logger.info(f"  Feature cache not found – running LIWC pipeline for {dataset_name} …")
            return run_liwc_pipeline(dataset_name)
        logger.info(f"  Loading cached LIWC features for {dataset_name} …")
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
        feat_cols = [c for c in train_df.columns if c != "label"]
        X_train = train_df[feat_cols].values.astype(np.float64)
        y_train = train_df["label"].values.astype(int)
        X_test  = test_df[feat_cols].values.astype(np.float64)
        y_test  = test_df["label"].values.astype(int)
        scaler  = joblib.load(str(scaler_path)) if scaler_path.exists() else None
        return {
            "dataset_name":  dataset_name,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "scaler":  scaler,
            "feature_names": feat_cols,
            "mode": "LIWC-Approx",
        }
    elif mode == "tfidf":
        return run_tfidf_pipeline(dataset_name)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'liwc' or 'tfidf'.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature extraction pipeline.")
    p.add_argument("--mode",   choices=["liwc", "tfidf"], default="liwc",
                   help="Feature extraction mode (default: liwc)")
    p.add_argument("--datasets", nargs="+", default=DATASET_NAMES,
                   help="Which datasets to process")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _ensure_dirs()

    for ds in args.datasets:
        if args.mode == "liwc":
            run_liwc_pipeline(ds)
        else:
            run_tfidf_pipeline(ds)

    logger.info("\nFeature extraction complete.")
