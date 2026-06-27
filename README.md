# 🛡️ Malicious URL Classification

An end-to-end **Machine Learning** project that detects whether a URL is **Benign (0)** or **Malicious (1)**. The project covers the complete machine learning lifecycle, including data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, model persistence, and deployment through an interactive **Streamlit** web application.

---

## 📌 Project Overview

Cybersecurity threats often originate from malicious URLs used in phishing, malware distribution, spam campaigns, and credential theft. This project applies Natural Language Processing (NLP) and Machine Learning techniques to automatically classify URLs before users visit them.

The pipeline automatically:

* Detects the URL and label columns
* Cleans and preprocesses the dataset
* Normalizes labels into binary classes
* Extracts handcrafted URL features
* Builds CountVectorizer and TF-IDF representations
* Handles class imbalance using SMOTE (when required)
* Trains and compares multiple machine learning models
* Performs hyperparameter tuning
* Evaluates model performance
* Saves the best trained model
* Deploys the classifier using Streamlit

---

# ✨ Features

* ✅ Automatic dataset schema detection
* ✅ Dataset profiling and preprocessing
* ✅ URL feature engineering
* ✅ Character and word n-gram vectorization
* ✅ CountVectorizer & TF-IDF comparison
* ✅ Class imbalance handling with SMOTE
* ✅ Multiple Machine Learning classifiers
* ✅ Cross-validation
* ✅ GridSearchCV & RandomizedSearchCV
* ✅ ROC, Precision-Recall and Confusion Matrix visualization
* ✅ Model persistence (.pkl)
* ✅ Interactive Streamlit application

---

# 📂 Project Structure

```text
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
│   └── screenshots/
│       ├── home.png
│       ├── malicious_warning.png
│       └── prediction.png
├── requirements.txt
└── src/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/YourUsername/Malicious_URL_Classification.git
cd Malicious_URL_Classification
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📊 Dataset

Place the dataset inside

```text
data/dataset.csv
```

The pipeline automatically detects the URL and label columns and prepares the dataset for training.

---

# 🚀 Training

Run

```bash
python -m src.train
```

The training pipeline automatically performs

* Dataset analysis
* Data cleaning
* Feature engineering
* Text vectorization
* Model comparison
* Hyperparameter tuning
* Performance evaluation
* Artifact generation

All generated files are saved inside the **outputs/** and **models/** folders.

---

# 🌐 Streamlit Web Application

Launch the application

```bash
streamlit run app.py
```

The application allows users to

* Enter any URL
* Predict whether it is Benign or Malicious
* Display prediction confidence
* Inspect extracted URL features

---

# 📈 Model Performance

## Confusion Matrix

The confusion matrix illustrates how accurately the best-performing classifier distinguishes between benign and malicious URLs.

<p align="center">
<img src="outputs/figures/confusion_best_model.png" width="700">
</p>

---

## ROC Curve

The ROC Curve demonstrates the classifier's ability to separate malicious URLs from legitimate ones across different decision thresholds.

<p align="center">
<img src="outputs/figures/roc_best_model.png" width="700">
</p>

---

## Precision-Recall Curve

The Precision-Recall Curve highlights the model's effectiveness on the malicious class, which is especially important for cybersecurity applications involving imbalanced datasets.

<p align="center">
<img src="outputs/figures/pr_best_model.png" width="700">
</p>

---

# 💻 Streamlit Application

## Home Interface

The application's homepage allows users to enter any URL for analysis.

<p align="center">
<img src="outputs/screenshots/home.png" width="900">
</p>

---

## URL Prediction

Example of a successful prediction displaying the predicted class along with the confidence score.

<p align="center">
<img src="outputs/screenshots/prediction.png" width="900">
</p>

---

## Malicious URL Detection

Example showing the warning displayed when the model identifies a malicious URL.

<p align="center">
<img src="outputs/screenshots/malicious_warning.png" width="900">
</p>

---

# 📁 Generated Outputs

```
outputs/
├── figures/
├── metrics/
├── reports/
└── screenshots/
```

The project automatically generates:

* Model comparison tables
* Vectorizer comparison tables
* ROC Curve
* Precision-Recall Curve
* Confusion Matrix
* Classification reports
* Best model summary
* Streamlit screenshots

---

# 📦 Saved Models

```
models/
├── best_model.pkl
├── selected_vectorizer.pkl
└── tfidf_vectorizer.pkl
```

These artifacts are used directly by the Streamlit application for inference.

