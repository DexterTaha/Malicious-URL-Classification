"""Prediction utilities for the saved malicious URL classifier."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import extract_feature_frame, load_artifact, normalize_url


class URLPredictor:
    """Wrapper around the persisted best model."""

    def __init__(self, model_path: str | Path):
        self.model = load_artifact(model_path)

    def predict(self, url: str):
        """Predict class and probability for a single URL."""
        url = normalize_url(url)
        prediction = int(self.model.predict([url])[0])
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba([url])[0][1])
        elif hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function([url])[0])
            probability = 1.0 / (1.0 + np.exp(-score))
        else:
            probability = float(prediction)
        return prediction, probability, extract_feature_frame([url]).iloc[0].to_dict()
