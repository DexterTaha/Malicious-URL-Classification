"""Utility helpers for the malicious URL classification project."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

RANDOM_STATE = 42
SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "secure",
    "update",
    "verify",
    "bank",
    "paypal",
    "account",
    "confirm",
    "password",
    "admin",
    "payment",
    "wallet",
    "free",
    "bonus",
    "gift",
    "crypto",
    "bitcoin",
    "token",
    "support",
    "authentication",
]

LABEL_MAP = {
    "good": 0,
    "benign": 0,
    "safe": 0,
    "normal": 0,
    "legit": 0,
    "legitimate": 0,
    "bad": 1,
    "malicious": 1,
    "phishing": 1,
    "spam": 1,
    "suspect": 1,
    "1": 1,
    "0": 0,
}


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_artifact(obj: Any, path: str | Path) -> None:
    """Persist an object with joblib."""
    path = Path(path)
    ensure_directory(path.parent)
    joblib.dump(obj, path)


def load_artifact(path: str | Path) -> Any:
    """Load a persisted object with joblib."""
    return joblib.load(Path(path))


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Detect URL and target columns from a dataframe."""
    columns = {str(col).strip().lower(): col for col in df.columns}
    url_candidates = ["url", "urls", "link", "links", "website", "websites"]
    label_candidates = ["class", "label", "target", "y", "type", "category", "malicious"]

    url_col = next((columns[name] for name in url_candidates if name in columns), None)
    label_col = next((columns[name] for name in label_candidates if name in columns), None)

    if url_col is None:
        url_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    return url_col, label_col


def normalize_url(value: Any) -> str:
    """Convert URL-like values to a cleaned lowercase string."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("\n", "").replace("\r", "")
    text = re.sub(r"\s+", "", text)
    text = text.lower()
    if not re.match(r"^[a-z]+://", text):
        text = "http://" + text
    return text


def normalize_label(value: Any) -> int | None:
    """Map labels to binary 0/1 values when possible."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in LABEL_MAP:
        return int(LABEL_MAP[text])
    if text.isdigit() and text in {"0", "1"}:
        return int(text)
    return None


def safe_parse_url(url: str):
    """Parse a URL and return a ParseResult, tolerating malformed inputs."""
    parsed = urlparse(url)
    if not parsed.netloc and parsed.path:
        parsed = urlparse("http://" + url.lstrip("/"))
    return parsed


def compute_entropy(text: str) -> float:
    """Compute Shannon entropy for a string."""
    if not text:
        return 0.0
    counts = np.array(list(pd.Series(list(text)).value_counts()), dtype=float)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def has_ip_address(hostname: str) -> int:
    """Return 1 when hostname looks like an IP address."""
    return int(bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", hostname or "")))


def count_subdomains(hostname: str) -> int:
    """Count subdomain levels in a hostname."""
    if not hostname:
        return 0
    parts = [part for part in hostname.split(".") if part]
    return max(len(parts) - 2, 0)


def extract_tld(hostname: str) -> str:
    """Extract a simple TLD from a hostname."""
    if not hostname or "." not in hostname:
        return "unknown"
    return hostname.rsplit(".", 1)[-1].lower()


def url_character_statistics(url: str) -> dict[str, float | int | str]:
    """Generate handcrafted URL features."""
    normalized = normalize_url(url)
    parsed = safe_parse_url(normalized)
    hostname = parsed.netloc.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    url_body = normalized

    letters = [ch for ch in url_body if ch.isalpha()]
    digits = [ch for ch in url_body if ch.isdigit()]
    special = [ch for ch in url_body if not ch.isalnum()]
    uppercase = [ch for ch in str(url) if ch.isupper()]

    suspicious_hits = 0
    lowered = url_body.lower()
    for word in SUSPICIOUS_WORDS:
        if word in lowered:
            suspicious_hits += 1

    domain = hostname.split(":")[0]
    path_length = len(path)
    query_length = len(query)
    domain_length = len(domain)

    return {
        "url_length": len(url_body),
        "num_dots": url_body.count("."),
        "num_slashes": url_body.count("/"),
        "num_hyphens": url_body.count("-"),
        "num_underscores": url_body.count("_"),
        "num_digits": len(digits),
        "num_special_chars": len(special),
        "num_subdomains": count_subdomains(domain),
        "path_length": path_length,
        "query_length": query_length,
        "domain_length": domain_length,
        "tld": extract_tld(domain),
        "https_flag": int(normalized.startswith("https://")),
        "ip_address_flag": has_ip_address(domain),
        "suspicious_word_flag": int(suspicious_hits > 0),
        "suspicious_word_count": suspicious_hits,
        "entropy_score": compute_entropy(url_body),
        "digit_ratio": len(digits) / max(len(url_body), 1),
        "uppercase_ratio": len(uppercase) / max(len(str(url)), 1),
        "special_char_ratio": len(special) / max(len(url_body), 1),
        "letter_ratio": len(letters) / max(len(url_body), 1),
    }


def extract_feature_frame(urls: Iterable[str]) -> pd.DataFrame:
    """Convert a collection of URLs into a feature dataframe."""
    rows = [url_character_statistics(url) for url in urls]
    return pd.DataFrame(rows)


def coerce_binary_labels(series: pd.Series) -> pd.Series:
    """Convert a label series to 0/1 integers and drop unknowns."""
    return series.map(normalize_label)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a JSON file with UTF-8 encoding."""
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def timed_call(func, *args, **kwargs):
    """Execute a callable and return its result plus elapsed seconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed
