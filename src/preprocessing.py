"""Data loading, cleaning, splitting, and balancing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from .utils import coerce_binary_labels, detect_columns, ensure_directory, normalize_url


@dataclass
class DatasetBundle:
    """Container for cleaned dataset components."""

    dataframe: pd.DataFrame
    url_column: str
    label_column: str


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset with defensive encoding handling."""
    csv_path = Path(csv_path)
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Return a compact dataset analysis summary."""
    url_col, label_col = detect_columns(df)
    summary = {
        "shape": df.shape,
        "samples": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "unique_urls": int(df[url_col].astype(str).nunique()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "label_distribution": df[label_col].astype(str).value_counts(dropna=False).to_dict(),
        "url_column": url_col,
        "label_column": label_col,
    }
    return summary


def clean_dataset(df: pd.DataFrame) -> DatasetBundle:
    """Clean and normalize the dataset."""
    url_col, label_col = detect_columns(df)
    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=[url_col, label_col])
    cleaned[url_col] = cleaned[url_col].astype(str).map(normalize_url)
    cleaned[label_col] = coerce_binary_labels(cleaned[label_col])
    cleaned = cleaned.dropna(subset=[label_col])
    cleaned[label_col] = cleaned[label_col].astype(int)
    cleaned = cleaned[cleaned[url_col].str.len() > 0]
    cleaned = cleaned.drop_duplicates(subset=[url_col])
    cleaned = cleaned.reset_index(drop=True)
    return DatasetBundle(dataframe=cleaned, url_column=url_col, label_column=label_col)


def split_train_test(df: pd.DataFrame, url_column: str, label_column: str, test_size: float = 0.2):
    """Split the cleaned dataframe into train and test sets."""
    X = df[url_column].astype(str)
    y = df[label_column].astype(int)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )


def apply_smote(X_train_features, y_train):
    """Apply SMOTE to dense feature arrays when class imbalance exists."""
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train_features, y_train)


def save_class_distribution_plots(y_before, y_after, output_dir: str | Path) -> None:
    """Save before/after balancing bar charts."""
    output_dir = ensure_directory(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x=y_before, ax=axes[0], palette="viridis")
    axes[0].set_title("Before Balancing")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    sns.countplot(x=y_after, ax=axes[1], palette="magma")
    axes[1].set_title("After Balancing")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "class_balance_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_missing_values_heatmap(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Save a missing-values heatmap for the dataframe."""
    output_dir = ensure_directory(output_dir)
    fig = plt.figure(figsize=(12, 4))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "missing_values_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
