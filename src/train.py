"""Model training and selection pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from .feature_engineering import URLFeatureEngineer, handcrafted_feature_columns
    from .preprocessing import analyze_dataset, clean_dataset, load_dataset, split_train_test
    from .utils import RANDOM_STATE, ensure_directory, save_artifact, save_json
except ImportError:  # pragma: no cover
    from src.feature_engineering import URLFeatureEngineer, handcrafted_feature_columns
    from src.preprocessing import analyze_dataset, clean_dataset, load_dataset, split_train_test
    from src.utils import RANDOM_STATE, ensure_directory, save_artifact, save_json

MAX_TRAINING_SAMPLES = 300


@dataclass
class ModelResult:
    """Summary of a tuned model run."""

    name: str
    model: Any
    vectorizer_name: str
    best_params: dict[str, Any]
    best_score: float
    training_time: float
    testing_time: float
    metrics: dict[str, float]
    cv_scores: list[float]
    y_test_pred: np.ndarray
    y_test_proba: np.ndarray
    fitted_search: Any


def _to_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _to_sparse(matrix):
    if sparse.issparse(matrix):
        return matrix
    return sparse.csr_matrix(np.asarray(matrix))


class SparseTransformer(FunctionTransformer):
    """Convert dense arrays to sparse matrices for feature unions."""

    def __init__(self):
        super().__init__(func=_to_sparse, validate=False)


class DenseTransformer(FunctionTransformer):
    """Convert sparse matrices to dense arrays when required."""

    def __init__(self):
        super().__init__(func=_to_dense, validate=False)


def build_vectorizers():
    """Create the candidate text vectorizers."""
    return {
        "count_char": CountVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True),
        "count_word": CountVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True),
        "tfidf_char": TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True),
        "tfidf_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True),
    }


def build_handcrafted_pipeline():
    """Create the numeric/categorical handcrafted feature pipeline."""
    numeric_columns, categorical_columns = handcrafted_feature_columns()
    return Pipeline(
        steps=[
            ("features", URLFeatureEngineer()),
            (
                "encode",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                                    ("sparse", SparseTransformer()),
                                ]
                            ),
                            numeric_columns,
                        ),
                        (
                            "tld",
                            OneHotEncoder(handle_unknown="ignore"),
                            categorical_columns,
                        ),
                    ],
                    remainder="drop",
                    sparse_threshold=0.3,
                ),
            ),
        ]
    )


def build_feature_union(vectorizer):
    """Combine text vectorization and handcrafted features."""
    return FeatureUnion(
        transformer_list=[
            ("text", vectorizer),
            ("handcrafted", build_handcrafted_pipeline()),
        ]
    )


def build_models():
    """Return supported models and their parameter grids."""
    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver="saga"),
            {
                "model__C": [0.5, 1.0],
                "model__penalty": ["l2"],
            },
            False,
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [None, 10],
                "model__min_samples_split": [2, 5],
            },
            False,
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [50, 100],
                "model__max_depth": [None, 20],
            },
            False,
        ),
        "svm": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "model__C": [0.5, 1.0],
                "model__kernel": ["linear", "rbf"],
            },
            False,
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5],
                "model__weights": ["uniform", "distance"],
            },
            False,
        ),
        "naive_bayes": (
            MultinomialNB(),
            {
                "model__alpha": [0.1, 1.0],
            },
            False,
        ),
    }

    return models


def build_pipeline(vectorizer, model, requires_dense: bool, use_smote: bool):
    """Build a full modeling pipeline."""
    steps = [("features", build_feature_union(vectorizer))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)))
    if requires_dense:
        steps.append(("dense", DenseTransformer()))
    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def build_baseline_pipeline(vectorizer):
    """Build a lightweight pipeline for vectorizer comparison."""
    return build_pipeline(
        vectorizer,
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver="saga", n_jobs=-1),
        False,
        True,
    )


def score_estimator(estimator, X_test, y_test):
    """Compute standard classification metrics."""
    y_pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(X_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
        scores = np.asarray(scores, dtype=float)
        y_proba = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-9)
    else:
        y_proba = np.asarray(y_pred, dtype=float)

    return (
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        },
        y_pred,
        y_proba,
    )


def _count_param_space(param_grid: dict[str, list[Any]]) -> int:
    return int(prod(len(values) for values in param_grid.values()))


def fit_with_search(pipeline, param_grid, X_train, y_train):
    """Run grid or randomized search depending on grid size."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    total_combinations = _count_param_space(param_grid)
    if total_combinations > 8:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=min(8, total_combinations),
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            random_state=RANDOM_STATE,
        )
    else:
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

    start = perf_counter()
    search.fit(X_train, y_train)
    training_time = perf_counter() - start
    return search, training_time


def tune_single_model(name, pipeline, param_grid, X_train, y_train, X_test, y_test):
    """Run hyperparameter tuning for a single model."""
    search, training_time = fit_with_search(pipeline, param_grid, X_train, y_train)
    test_start = perf_counter()
    metrics, y_pred, y_proba = score_estimator(search.best_estimator_, X_test, y_test)
    testing_time = perf_counter() - test_start
    return ModelResult(
        name=name,
        model=search.best_estimator_,
        vectorizer_name="combined",
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        training_time=training_time,
        testing_time=testing_time,
        metrics=metrics,
        cv_scores=[float(search.best_score_)],
        y_test_pred=y_pred,
        y_test_proba=y_proba,
        fitted_search=search,
    )


def _sample_training_set(X_train: pd.Series, y_train: pd.Series, max_samples: int = MAX_TRAINING_SAMPLES):
    """Create a stratified subsample for model selection when needed."""
    if len(X_train) <= max_samples:
        return X_train, y_train
    X_small, _, y_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_samples,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    return X_small, y_small


def compare_vectorizers(X_train, y_train, X_test, y_test):
    """Benchmark CountVectorizer and TF-IDF variants using a shared baseline model."""
    vectorizers = build_vectorizers()
    rows = []
    for name, vectorizer in vectorizers.items():
        pipeline = build_baseline_pipeline(vectorizer)
        pipeline.fit(X_train, y_train)
        metrics, _, _ = score_estimator(pipeline, X_test, y_test)
        rows.append({"vectorizer": name, **metrics})
    comparison = pd.DataFrame(rows).sort_values(by=["f1", "roc_auc", "accuracy"], ascending=False)
    return comparison, comparison.iloc[0]["vectorizer"]


def train_models(X_train, y_train, X_test, y_test, output_dir, vectorizer_name: str):
    """Train and tune all supported models on the selected vectorizer."""
    output_dir = ensure_directory(output_dir)
    models = build_models()
    results: list[ModelResult] = []

    X_train_sample, y_train_sample = _sample_training_set(X_train, y_train)
    vectorizer = build_vectorizers()[vectorizer_name]

    for model_name, (model, param_grid, requires_dense) in models.items():
        print(f"Training {model_name} with {vectorizer_name}...", flush=True)
        pipeline = build_pipeline(vectorizer, model, requires_dense, True)
        result = tune_single_model(
            f"{model_name}__{vectorizer_name}",
            pipeline,
            param_grid,
            X_train_sample,
            y_train_sample,
            X_test,
            y_test,
        )
        result.vectorizer_name = vectorizer_name
        results.append(result)

    results.sort(key=lambda item: (item.metrics["f1"], item.metrics["roc_auc"], item.metrics["accuracy"]), reverse=True)
    best = results[0]

    models_dir = ensure_directory(Path(output_dir))
    save_artifact(best.model, models_dir / "best_model.pkl")

    vectorizer = best.model.named_steps["features"].transformer_list[0][1]
    save_artifact(vectorizer, models_dir / "tfidf_vectorizer.pkl")

    save_json(
        {
            "best_model": best.name,
            "best_params": best.best_params,
            "best_score": best.best_score,
            "metrics": best.metrics,
        },
        Path(output_dir).parent / "outputs" / "reports" / "best_model_summary.json",
    )

    return results, best


def run_training_pipeline(csv_path: str | Path, project_root: str | Path):
    """Load, clean, split, train, and persist all project artifacts."""
    project_root = Path(project_root)
    outputs_dir = project_root / "outputs"
    figures_dir = ensure_directory(outputs_dir / "figures")
    reports_dir = ensure_directory(outputs_dir / "reports")
    metrics_dir = ensure_directory(outputs_dir / "metrics")
    models_dir = ensure_directory(project_root / "models")

    raw_df = load_dataset(csv_path)
    analysis = analyze_dataset(raw_df)
    cleaned_bundle = clean_dataset(raw_df)
    cleaned_df = cleaned_bundle.dataframe

    X_train, X_test, y_train, y_test = split_train_test(
        cleaned_df,
        cleaned_bundle.url_column,
        cleaned_bundle.label_column,
    )

    vectorizer_sample_X, vectorizer_sample_y = _sample_training_set(X_train, y_train)
    vectorizer_comparison, best_vectorizer_name = compare_vectorizers(
        vectorizer_sample_X,
        vectorizer_sample_y,
        X_test,
        y_test,
    )
    vectorizer_comparison.to_csv(metrics_dir / "vectorizer_comparison.csv", index=False)

    results, best = train_models(X_train, y_train, X_test, y_test, models_dir, best_vectorizer_name)

    try:
        from .evaluate import (
            build_metrics_table,
            learning_curve_plot,
            plot_confusion_matrix,
            plot_precision_recall_curve,
            plot_roc_curve,
        )
    except ImportError:  # pragma: no cover
        from src.evaluate import (
            build_metrics_table,
            learning_curve_plot,
            plot_confusion_matrix,
            plot_precision_recall_curve,
            plot_roc_curve,
        )

    metrics_table = build_metrics_table(results)
    metrics_table.to_csv(metrics_dir / "model_comparison.csv", index=False)
    save_json(metrics_table.to_dict(orient="records")[0], reports_dir / "best_model_record.json")

    learning_X, learning_y = _sample_training_set(X_train, y_train, max_samples=min(len(X_train), 3000))

    for result in results[:3]:
        model_label = result.name.replace("__", "_")
        plot_confusion_matrix(y_test, result.y_test_pred, figures_dir / f"confusion_{model_label}.png", f"Confusion Matrix - {result.name}")
        plot_roc_curve(y_test, result.y_test_proba, figures_dir / f"roc_{model_label}.png", f"ROC Curve - {result.name}")
        plot_precision_recall_curve(y_test, result.y_test_proba, figures_dir / f"pr_{model_label}.png", f"PR Curve - {result.name}")

    learning_curve_plot(best.model, learning_X, learning_y, figures_dir / f"learning_{best.name.replace('__', '_')}.png", f"Learning Curve - {best.name}")

    return {
        "analysis": analysis,
        "cleaned_rows": len(cleaned_df),
        "vectorizer_comparison": vectorizer_comparison,
        "results": results,
        "best": best,
        "metrics_table": metrics_table,
    }


def main():
    """CLI entry point for model training."""
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "dataset.csv"
    run_training_pipeline(csv_path, project_root)


if __name__ == "__main__":
    main()