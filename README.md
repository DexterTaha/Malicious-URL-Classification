# Malicious URL Classification

End-to-end machine learning project for classifying URLs as benign (0) or malicious (1). The project includes automated dataset analysis, cleaning, handcrafted feature engineering, text vectorization, class-imbalance handling, model selection, hyperparameter tuning, evaluation, model persistence, and a Streamlit inference app.

## Project Overview

The pipeline loads a CSV dataset, detects the URL and label columns automatically when possible, normalizes labels to a binary target, engineers URL-level features, compares text representations, trains multiple classifiers, and saves the best performing pipeline as a reusable artifact.

## Features

- Automatic schema detection for URL and label columns
- Dataset profiling and visualization
- Cleaning and label normalization
- Handcrafted URL features
- CountVectorizer and TF-IDF with character and word n-grams
- Class imbalance handling with SMOTE when needed
- Multiple model training with cross-validation
- GridSearchCV and RandomizedSearchCV tuning
- Evaluation metrics, curves, and comparison tables
- Saved best model and selected text vectorizer
- Streamlit web app for live URL prediction

## Project Structure

```
Malicious-URL-Classification/
├── README.md
├── app.py
├── build_artifacts.py
├── data/
│   └── dataset.csv
├── models/
│   ├── best_model.pkl
│   ├── selected_vectorizer.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── malicious_url_classification.ipynb
├── outputs/
│   ├── figures/
│   │   ├── confusion_best_model.png
│   │   ├── pr_best_model.png
│   │   └── roc_best_model.png
│   ├── metrics/
│   │   ├── model_comparison.csv
│   │   └── vectorizer_comparison.csv
│   ├── reports/
│   │   ├── best_model_summary.json
│   │   └── sentinel.txt
│   └── screenshots/
│       ├── home.png
│       ├── malicious_warning.png
│       └── prediction.png
├── requirements.txt
└── src/
    ├── __init__.py
    ├── __pycache__/
    │   ├── __init__.cpython-314.pyc
    │   ├── evaluate.cpython-314.pyc
    │   ├── feature_engineering.cpython-314.pyc
    │   ├── predict.cpython-314.pyc
    │   ├── preprocessing.cpython-314.pyc
    │   ├── train.cpython-314.pyc
    │   └── utils.cpython-314.pyc
    ├── evaluate.py
    ├── feature_engineering.py
    ├── predict.py
    ├── preprocessing.py
    ├── train.py
    └── utils.py

```

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Place the CSV file at:

```text
Malicious_URL_Classification/data/dataset.csv
```

The included notebook and training script will automatically detect the URL and label columns when the schema matches the expected two-column structure.

## Training

Run the training pipeline from the project root:

```bash
python -m src.train
```

This will:

- analyze the dataset
- clean and normalize it
- compare vectorizer families
- train and tune multiple classifiers
- evaluate all models
- save the best model artifacts
- write metrics and figures to `outputs/`

## Evaluation

Generated outputs are stored in:

- `outputs/metrics/`
- `outputs/reports/`
- `outputs/figures/`

Key artifacts include:

- model comparison table
- vectorizer comparison table
- confusion matrices
- ROC curves
- precision-recall curves
- learning curves
- best model summary

## Streamlit App

Launch the web app with:

```bash
streamlit run app.py
```

The interface allows you to paste a URL, submit a prediction, inspect the predicted class, view the probability score, and review extracted URL features.

## Notebook

Open `notebooks/malicious_url_classification.ipynb` for the full end-to-end workflow with markdown explanations and executable cells.

## Example Screenshots

Add screenshots here after running the Streamlit app:

- `outputs/screenshots/home.png`
- `outputs/screenshots/prediction.png`
- `outputs/screenshots/malicious_warning.png`

## Requirements

See `requirements.txt` for the full dependency list.

## Future Improvements

- Add calibration plots for probability reliability
- Expand feature extraction with domain reputation signals
- Add model explainability with SHAP or permutation importance
- Support incremental retraining on updated datasets
- Expose a batch prediction endpoint
