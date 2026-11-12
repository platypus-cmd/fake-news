"""
Fake News Detection - Data Preprocessing Module
Replication study based on Ahmad et al. (2020)

This module handles data preprocessing for:
- DS1: ISOT Fake News Dataset (Fake.csv, True.csv)
- DS2: Kaggle Fake News Dataset (train.csv)
- DS3: Kaggle Fake News Detection Dataset (data.csv)
- DS4: Combined master dataset (DS1 + DS2 + DS3 shuffled)

Standardized Label Conventions:
- 0: Fake / Unreliable
- 1: Truthful / Real / Reliable
"""

import os
import re
import html
import logging
from typing import Dict, Tuple, Optional
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def clean_text(text: Optional[str]) -> str:
    """
    Clean text by:
    1. Unescaping HTML entities (e.g. &amp;, &quot;)
    2. Stripping HTML tags (e.g. <p>, <br>, <img ...>)
    3. Normalizing whitespace (replacing multiple spaces/newlines with a single space)
    4. Stripping leading and trailing whitespace
    """
    if text is None or not isinstance(text, str):
        return ""
    # Unescape HTML entities
    text = html.unescape(text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_by_word_count(df: pd.DataFrame, text_col: str = "text", min_words: int = 20) -> pd.DataFrame:
    """
    Filter out articles where word count in text_col is strictly less than min_words.
    """
    word_counts = df[text_col].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    filtered_df = df[word_counts >= min_words].copy()
    dropped_count = len(df) - len(filtered_df)
    logger.info(f"Filtered {dropped_count:,} articles with fewer than {min_words} words. Remaining: {len(filtered_df):,}")
    return filtered_df


def preprocess_ds1(raw_dir: str = "data/raw", min_words: int = 20) -> pd.DataFrame:
    """
    Preprocess DS1 (ISOT dataset: Fake.csv, True.csv).
    - Assigns label = 0 to Fake, label = 1 to True.
    - Merges headline (title) and body (text).
    - Strips HTML tags and normalizes whitespace.
    - Removes non-content columns (subject, date).
    - Filters out articles with combined length < min_words.
    - Returns DataFrame with standardized ['text', 'label'].
    """
    fake_path = os.path.join(raw_dir, "Fake.csv")
    true_path = os.path.join(raw_dir, "True.csv")
    logger.info(f"Loading DS1 raw files: {fake_path} and {true_path}")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Assign binary labels: 0 for Fake, 1 for True
    fake_df["label"] = 0
    true_df["label"] = 1

    ds1 = pd.concat([fake_df, true_df], ignore_index=True)
    logger.info(f"DS1 raw loaded: {len(ds1):,} rows (Fake: {len(fake_df):,}, True: {len(true_df):,})")

    # Merge headline (title) and body (text) & clean
    headline = ds1["title"].fillna("").astype(str)
    body = ds1["text"].fillna("").astype(str)
    ds1["text"] = (headline + " " + body).apply(clean_text)

    # Keep only content ('text') and target ('label') columns
    ds1_cleaned = ds1[["text", "label"]].copy()

    # Filter out articles with fewer than min_words
    ds1_cleaned = filter_by_word_count(ds1_cleaned, text_col="text", min_words=min_words)
    ds1_cleaned.reset_index(drop=True, inplace=True)
    return ds1_cleaned


def preprocess_ds2(raw_dir: str = "data/raw", min_words: int = 20) -> pd.DataFrame:
    """
    Preprocess DS2 (Kaggle Fake News: train.csv).
    - Ignores test.csv (lacks ground-truth labels).
    - Standardizes labels (0 for Fake, 1 for Truthful).
      Note: Raw Kaggle train.csv uses 1 for unreliable (Fake) and 0 for reliable (Truthful).
      Mapping: 1 -> 0 (Fake), 0 -> 1 (Truthful).
    - Merges headline (title) and body (text).
    - Strips HTML tags and normalizes whitespace.
    - Removes non-content columns (author, id).
    - Filters out articles with combined length < min_words.
    - Returns DataFrame with standardized ['text', 'label'].
    """
    train_path = os.path.join(raw_dir, "train.csv")
    logger.info(f"Loading DS2 raw file: {train_path}")

    ds2 = pd.read_csv(train_path)
    logger.info(f"DS2 raw loaded: {len(ds2):,} rows")

    # Standardize labels (0 for Fake, 1 for Truthful)
    # Raw labels: 1 = Unreliable (Fake), 0 = Reliable (Truthful)
    ds2["label"] = ds2["label"].map({1: 0, 0: 1})

    # Merge headline (title) and body (text) & clean
    headline = ds2["title"].fillna("").astype(str)
    body = ds2["text"].fillna("").astype(str)
    ds2["text"] = (headline + " " + body).apply(clean_text)

    # Keep only content ('text') and target ('label') columns
    ds2_cleaned = ds2[["text", "label"]].copy()

    # Filter out articles with fewer than min_words
    ds2_cleaned = filter_by_word_count(ds2_cleaned, text_col="text", min_words=min_words)
    ds2_cleaned.reset_index(drop=True, inplace=True)
    return ds2_cleaned


def preprocess_ds3(raw_dir: str = "data/raw", min_words: int = 20) -> pd.DataFrame:
    """
    Preprocess DS3 (Kaggle Fake News Detection: data.csv).
    - Converts text/numeric labels (FAKE -> 0, REAL -> 1).
    - Merges headline (Headline) and body (Body).
    - Strips HTML tags and normalizes whitespace.
    - Removes non-content columns (URLs).
    - Filters out articles with combined length < min_words.
    - Returns DataFrame with standardized ['text', 'label'].
    """
    data_path = os.path.join(raw_dir, "data.csv")
    logger.info(f"Loading DS3 raw file: {data_path}")

    ds3 = pd.read_csv(data_path)
    logger.info(f"DS3 raw loaded: {len(ds3):,} rows")

    # Convert text/numeric labels: FAKE -> 0, REAL -> 1
    label_mapping = {
        "FAKE": 0,
        "REAL": 1,
        0: 0,
        1: 1,
        "0": 0,
        "1": 1
    }
    ds3["label"] = ds3["Label"].map(label_mapping)

    # Merge headline (Headline) and body (Body) & clean
    headline = ds3["Headline"].fillna("").astype(str)
    body = ds3["Body"].fillna("").astype(str)
    ds3["text"] = (headline + " " + body).apply(clean_text)

    # Keep only content ('text') and target ('label') columns
    ds3_cleaned = ds3[["text", "label"]].copy()

    # Filter out articles with fewer than min_words
    ds3_cleaned = filter_by_word_count(ds3_cleaned, text_col="text", min_words=min_words)
    ds3_cleaned.reset_index(drop=True, inplace=True)
    return ds3_cleaned


def create_ds4(ds1: pd.DataFrame, ds2: pd.DataFrame, ds3: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Concatenate cleaned DS1, DS2, and DS3 into a single master dataframe.
    Shuffles with random_state=42 and resets index.
    """
    logger.info(f"Concatenating datasets: DS1 ({len(ds1):,}), DS2 ({len(ds2):,}), DS3 ({len(ds3):,})")
    ds4 = pd.concat([ds1, ds2, ds3], ignore_index=True)
    logger.info(f"Total concatenated rows before shuffle: {len(ds4):,}")

    ds4_shuffled = ds4.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    logger.info(f"Shuffled DS4 with random_state={random_state}")
    return ds4_shuffled


def print_summary(datasets: Dict[str, pd.DataFrame]) -> None:
    """
    Print final row counts, label distribution, and missing value check for all datasets.
    """
    print("\n" + "=" * 70)
    print("                 DATA PREPROCESSING SUMMARY")
    print("=" * 70)

    summary_rows = []
    for name, df in datasets.items():
        total_rows = len(df)
        missing_text = df["text"].isna().sum()
        missing_label = df["label"].isna().sum()
        total_missing = missing_text + missing_label

        val_counts = df["label"].value_counts().to_dict()
        fake_count = val_counts.get(0, 0)
        true_count = val_counts.get(1, 0)
        fake_pct = (fake_count / total_rows * 100) if total_rows > 0 else 0
        true_pct = (true_count / total_rows * 100) if total_rows > 0 else 0

        summary_rows.append({
            "Dataset": name,
            "Total Rows": f"{total_rows:,}",
            "Fake (0)": f"{fake_count:,} ({fake_pct:.1f}%)",
            "Truthful (1)": f"{true_count:,} ({true_pct:.1f}%)",
            "Missing Values": f"{total_missing:,}"
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    print("\nDetailed Missing Value Check:")
    for name, df in datasets.items():
        print(f"  [{name}] Missing text: {df['text'].isna().sum()} | Missing label: {df['label'].isna().sum()}")
    print("=" * 70 + "\n")


def run_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end autonomous data preprocessing pipeline:
    1. Preprocesses DS1, DS2, DS3.
    2. Exports individual cleaned CSVs.
    3. Creates, shuffles, and exports master DS4_combined.csv.
    4. Prints comprehensive summary statistics.
    """
    os.makedirs(processed_dir, exist_ok=True)

    # Preprocess individual datasets
    ds1_cleaned = preprocess_ds1(raw_dir=raw_dir)
    ds2_cleaned = preprocess_ds2(raw_dir=raw_dir)
    ds3_cleaned = preprocess_ds3(raw_dir=raw_dir)

    # Save individual cleaned datasets
    ds1_out = os.path.join(processed_dir, "DS1_cleaned.csv")
    ds2_out = os.path.join(processed_dir, "DS2_cleaned.csv")
    ds3_out = os.path.join(processed_dir, "DS3_cleaned.csv")

    logger.info(f"Saving DS1 to {ds1_out}")
    ds1_cleaned.to_csv(ds1_out, index=False)

    logger.info(f"Saving DS2 to {ds2_out}")
    ds2_cleaned.to_csv(ds2_out, index=False)

    logger.info(f"Saving DS3 to {ds3_out}")
    ds3_cleaned.to_csv(ds3_out, index=False)

    # Create and save master DS4
    ds4_combined = create_ds4(ds1_cleaned, ds2_cleaned, ds3_cleaned, random_state=random_state)
    ds4_out = os.path.join(processed_dir, "DS4_combined.csv")
    logger.info(f"Saving DS4 to {ds4_out}")
    ds4_combined.to_csv(ds4_out, index=False)

    datasets = {
        "DS1": ds1_cleaned,
        "DS2": ds2_cleaned,
        "DS3": ds3_cleaned,
        "DS4": ds4_combined
    }

    # Summary report
    print_summary(datasets)
    return datasets


if __name__ == "__main__":
    run_pipeline()
