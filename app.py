"""Streamlit app for malicious URL classification."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.predict import URLPredictor
from src.utils import extract_feature_frame

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

st.set_page_config(page_title="Malicious URL Classifier", layout="wide")
st.title("Malicious URL Classification")
st.write("Enter a URL to detect whether it is benign or malicious.")

@st.cache_resource
def load_predictor():
    return URLPredictor(MODEL_PATH)

predictor = load_predictor()
url = st.text_input("URL", placeholder="https://example.com/login")

if st.button("Predict"):
    if not url.strip():
        st.error("Please enter a URL.")
    else:
        prediction, probability, feature_map = predictor.predict(url)
        label = "Malicious" if prediction == 1 else "Benign"
        color = "red" if prediction == 1 else "green"
        confidence = probability if prediction == 1 else 1 - probability

        st.markdown(f"### Prediction: <span style='color:{color};'>{label}</span>", unsafe_allow_html=True)
        st.write(f"Entered URL: {url}")
        st.write(f"Probability score: {probability:.4f}")
        st.write(f"Confidence: {confidence:.4f}")
        st.progress(min(max(probability if prediction == 1 else 1 - probability, 0.0), 1.0))
        if prediction == 1:
            st.warning("This URL is predicted to be malicious.")
        st.subheader("Extracted URL Features")
        st.dataframe(extract_feature_frame([url]).T, use_container_width=True)
        st.subheader("Feature Details")
        st.json(feature_map)
