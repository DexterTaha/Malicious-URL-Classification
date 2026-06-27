"""Feature engineering helpers for malicious URL classification."""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .utils import extract_feature_frame


class URLFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for handcrafted URL features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        urls = pd.Series(X).astype(str)
        return extract_feature_frame(urls)


def handcrafted_feature_columns() -> tuple[list[str], list[str]]:
    """Return numeric and categorical handcrafted feature columns."""
    numeric_columns = [
        "url_length",
        "num_dots",
        "num_slashes",
        "num_hyphens",
        "num_underscores",
        "num_digits",
        "num_special_chars",
        "num_subdomains",
        "path_length",
        "query_length",
        "domain_length",
        "https_flag",
        "ip_address_flag",
        "suspicious_word_flag",
        "suspicious_word_count",
        "entropy_score",
        "digit_ratio",
        "uppercase_ratio",
        "special_char_ratio",
        "letter_ratio",
    ]
    categorical_columns = ["tld"]
    return numeric_columns, categorical_columns


def build_feature_dataframe(urls: Iterable[str]) -> pd.DataFrame:
    """Create a feature dataframe from iterable URLs."""
    return extract_feature_frame(urls)
