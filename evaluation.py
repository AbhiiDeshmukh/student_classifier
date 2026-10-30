"""Shared constants and metric helpers used by both the training script and the Streamlit app."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "students_dropout_academic_success.csv"
TEST_DATA_PATH = ROOT_DIR / "test_data.csv"
METRICS_PATH = MODEL_DIR / "test_metrics.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET_COL = "Target"
CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]

# Display name -> saved pipeline file inside model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The UCI file ships with stray whitespace/tabs in some headers."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def compute_metrics(y_true, y_pred, y_proba, classes) -> dict:
    """Six required metrics. Multi-class averages are support-weighted; AUC is one-vs-rest."""
    y_true = np.asarray(y_true)
    try:
        if len(classes) == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted", labels=list(classes)
            )
    except ValueError:
        # AUC is undefined when the evaluated data contains a single class
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_model(model, X: pd.DataFrame, y) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Returns (metrics, confusion matrix, classification report as a DataFrame)."""
    classes = list(model.classes_)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    metrics = compute_metrics(y, y_pred, y_proba, classes)
    cm = confusion_matrix(y, y_pred, labels=classes)
    report = classification_report(
        y, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    return metrics, cm, pd.DataFrame(report).transpose()
