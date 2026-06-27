from pathlib import Path

from sklearn.linear_model import LogisticRegression

from src.evaluate import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from src.preprocessing import clean_dataset, load_dataset, split_train_test
from src.train import build_pipeline, build_vectorizers, score_estimator
from src.utils import ensure_directory, save_artifact, save_json


def main():
    root = Path(__file__).resolve().parent
    models_dir = ensure_directory(root / 'models')
    figures_dir = ensure_directory(root / 'outputs' / 'figures')
    metrics_dir = ensure_directory(root / 'outputs' / 'metrics')
    reports_dir = ensure_directory(root / 'outputs' / 'reports')

    df = load_dataset(root / 'data' / 'dataset.csv')
    cleaned = clean_dataset(df)
    cleaned_df = cleaned.dataframe.sample(n=1000, random_state=42)
    X_train, X_test, y_train, y_test = split_train_test(cleaned_df, cleaned.url_column, cleaned.label_column)

    vectorizer = build_vectorizers()['count_char']
    model = LogisticRegression(max_iter=2000, solver='saga', random_state=42)
    pipeline = build_pipeline(vectorizer, model, False, True)
    pipeline.fit(X_train, y_train)
    metrics, y_pred, y_proba = score_estimator(pipeline, X_test, y_test)

    save_artifact(pipeline, models_dir / 'best_model.pkl')
    save_artifact(vectorizer, models_dir / 'tfidf_vectorizer.pkl')
    plot_confusion_matrix(y_test, y_pred, figures_dir / 'confusion_best_model.png', 'Confusion Matrix - Best Model')
    plot_roc_curve(y_test, y_proba, figures_dir / 'roc_best_model.png', 'ROC Curve - Best Model')
    plot_precision_recall_curve(y_test, y_proba, figures_dir / 'pr_best_model.png', 'PR Curve - Best Model')
    save_json({'best_model': 'logistic_regression__count_char', 'best_params': {'solver': 'saga', 'max_iter': 2000}, 'metrics': metrics}, reports_dir / 'best_model_summary.json')
    print(metrics)


if __name__ == '__main__':
    main()
