"""
Feature Extraction and Feature Scaling Module
Replication study based on Ahmad et al. (2020)

This module handles:
1. Loading cleaned datasets (DS1, DS2, DS3, DS4) from data/processed/.
2. Stratified train/test splitting (70% train, 30% test, random_state=42).
3. Text feature extraction using TfidfVectorizer(max_features=5000, ngram_range=(1, 2)).
4. Feature scaling using scikit-learn MinMaxScaler transformed into [0, 1].
5. Saving the fitted scaler and vectorizer objects using joblib, and sparse matrices with scipy.sparse.
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str, processed_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load a preprocessed dataset by name (DS1, DS2, DS3, DS4).
    """
    filename_map = {
        "DS1": "DS1_cleaned.csv",
        "DS2": "DS2_cleaned.csv",
        "DS3": "DS3_cleaned.csv",
        "DS4": "DS4_combined.csv"
    }
    filename = filename_map.get(dataset_name.upper())
    if not filename:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected one of {list(filename_map.keys())}")
    
    file_path = os.path.join(processed_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed dataset not found at {file_path}")
    
    logger.info(f"Loading {dataset_name} from {file_path}")
    df = pd.read_csv(file_path)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Perform 70/30 stratified train/test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )
    logger.info(
        f"Split data: Train={len(X_train):,} ({len(X_train)/len(df)*100:.1f}%), "
        f"Test={len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)"
    )
    return X_train, X_test, y_train, y_test


def fit_min_max_scaler_sparse(
    X_train: csr_matrix,
    X_test: Optional[csr_matrix] = None
) -> Tuple[csr_matrix, Optional[csr_matrix], MinMaxScaler]:
    """
    Fit a scikit-learn MinMaxScaler on sparse feature matrix:
    - Calculates feature minimums (0 for TF-IDF word vectors) and maximums.
    - Scales values into [0, 1] while preserving sparsity.
    - Populates a real scikit-learn MinMaxScaler instance for export with joblib.
    """
    n_samples, n_features = X_train.shape

    data_min = np.zeros(n_features, dtype=np.float64)
    data_max = np.array(X_train.max(axis=0).toarray()).ravel().astype(np.float64)
    data_range = np.where(data_max == 0.0, 1.0, data_max - data_min)
    scale = 1.0 / data_range

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_ = data_min
    scaler.data_max_ = data_max
    scaler.data_range_ = data_range
    scaler.scale_ = scale
    scaler.min_ = np.zeros(n_features, dtype=np.float64)
    scaler.n_features_in_ = n_features
    scaler.n_samples_seen_ = n_samples

    X_train_scaled = X_train.multiply(scale).tocsr()
    X_test_scaled = X_test.multiply(scale).tocsr() if X_test is not None else None

    return X_train_scaled, X_test_scaled, scaler


def extract_and_scale_features(
    dataset_name: str,
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    save_artifacts: bool = True,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    End-to-end feature extraction and scaling pipeline for a dataset.
    Caches processed sparse feature matrices for instant loading.
    """
    os.makedirs(models_dir, exist_ok=True)
    ds_key = dataset_name.lower()

    train_npz = os.path.join(models_dir, f"X_train_{ds_key}.npz")
    test_npz = os.path.join(models_dir, f"X_test_{ds_key}.npz")
    ytrain_npy = os.path.join(models_dir, f"y_train_{ds_key}.npy")
    ytest_npy = os.path.join(models_dir, f"y_test_{ds_key}.npy")
    scaler_path = os.path.join(models_dir, f"scaler_{ds_key}.joblib")
    vec_path = os.path.join(models_dir, f"vectorizer_{ds_key}.joblib")

    # Fast cache check
    if (
        use_cache
        and os.path.exists(train_npz)
        and os.path.exists(test_npz)
        and os.path.exists(ytrain_npy)
        and os.path.exists(ytest_npy)
        and os.path.exists(scaler_path)
        and os.path.exists(vec_path)
    ):
        logger.info(f"Loading cached features and labels for {dataset_name}...")
        X_train_scaled = load_npz(train_npz)
        X_test_scaled = load_npz(test_npz)
        y_train = np.load(ytrain_npy)
        y_test = np.load(ytest_npy)
        scaler = joblib.load(scaler_path)
        vectorizer = joblib.load(vec_path)
        return {
            "dataset_name": dataset_name,
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "vectorizer": vectorizer,
            "scaler": scaler
        }

    df = load_dataset(dataset_name, processed_dir=processed_dir)
    X_train_raw, X_test_raw, y_train_s, y_test_s = split_data(df)
    y_train = y_train_s.values
    y_test = y_test_s.values

    # Check if vectorizer & scaler exist
    if os.path.exists(vec_path) and os.path.exists(scaler_path):
        logger.info(f"Loading existing vectorizer from {vec_path}...")
        vectorizer = joblib.load(vec_path)
        scaler = joblib.load(scaler_path)
        X_train_tfidf = vectorizer.transform(X_train_raw)
        X_test_tfidf = vectorizer.transform(X_test_raw)
        X_train_scaled = X_train_tfidf.multiply(scaler.scale_).tocsr()
        X_test_scaled = X_test_tfidf.multiply(scaler.scale_).tocsr()
    else:
        logger.info(f"Extracting TF-IDF features (max_features={max_features}, ngrams={ngram_range})...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        X_train_tfidf = vectorizer.fit_transform(X_train_raw)
        X_test_tfidf = vectorizer.transform(X_test_raw)

        logger.info("Applying MinMaxScaler to normalize feature values to [0, 1]...")
        X_train_scaled, X_test_scaled, scaler = fit_min_max_scaler_sparse(X_train_tfidf, X_test_tfidf)

        if save_artifacts:
            joblib.dump(scaler, scaler_path)
            joblib.dump(vectorizer, vec_path)

    if save_artifacts:
        save_npz(train_npz, X_train_scaled)
        save_npz(test_npz, X_test_scaled)
        np.save(ytrain_npy, y_train)
        np.save(ytest_npy, y_test)
        logger.info(f"Cached precomputed feature matrices for {dataset_name} to {models_dir}/")

    return {
        "dataset_name": dataset_name,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "vectorizer": vectorizer,
        "scaler": scaler
    }


if __name__ == "__main__":
    for ds in ["DS1", "DS2", "DS3", "DS4"]:
        logger.info(f"\n{'='*50}\nProcessing Feature Extraction for {ds}\n{'='*50}")
        extract_and_scale_features(ds)
