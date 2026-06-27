"""Evaluation helpers and report generation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import learning_curve, cross_val_score

from .utils import ensure_directory, save_json


def build_metrics_table(results):
    """Create a tabular summary from model results."""
    rows = []
    for result in results:
        row = {
            "model": result.name,
            "best_score": result.best_score,
            "training_time": result.training_time,
            "testing_time": result.testing_time,
            **result.metrics,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by=["f1", "roc_auc", "accuracy"], ascending=False)


def plot_confusion_matrix(y_true, y_pred, output_path: str | Path, title: str):
    """Save a confusion matrix heatmap."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, output_path: str | Path, title: str):
    """Save a ROC curve figure."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(y_true, y_proba, output_path: str | Path, title: str):
    """Save a precision-recall curve figure."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(metrics_table: pd.DataFrame, output_path: str | Path):
    """Save a model accuracy comparison chart."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    fig = plt.figure(figsize=(12, 5))
    sns.barplot(data=metrics_table, x="model", y="accuracy", palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_reports(results, output_dir: str | Path):
    """Persist figures and metrics for all trained models."""
    output_dir = ensure_directory(output_dir)
    figures_dir = ensure_directory(Path(output_dir).parent / "figures")
    metrics_table = build_metrics_table(results)
    metrics_table.to_csv(Path(output_dir) / "model_comparison.csv", index=False)
    save_json(metrics_table.to_dict(orient="records")[0], Path(output_dir) / "best_model_record.json")
    plot_accuracy_comparison(metrics_table, Path(figures_dir) / "accuracy_comparison.png")
    return metrics_table


def learning_curve_plot(estimator, X, y, output_path: str | Path, title: str):
    """Generate a learning curve plot."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    train_sizes, train_scores, valid_scores = learning_curve(estimator, X, y, cv=3, n_jobs=-1)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
    plt.plot(train_sizes, valid_scores.mean(axis=1), label="Validation score")
    plt.xlabel("Training Samples")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
