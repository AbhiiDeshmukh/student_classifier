"""
Train and evaluate the five classifiers required by ML Assignment 2.

Dataset : UCI "Predict Students' Dropout and Academic Success" (id 697)
Task    : 3-class classification of Target -> Dropout / Enrolled / Graduate

Usage   : python train_models.py
Outputs : model/*.joblib, model/test_metrics.csv, model/metadata.json, test_data.csv
"""

import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import (
    METADATA_PATH,
    METRIC_NAMES,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_DATA_PATH,
    clean_columns,
    evaluate_model,
)

SEED = 5361
TEST_SIZE = 0.20

# Integer-coded nominal attributes: the codes carry no order, so they are one-hot encoded.
NOMINAL_COLS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH, sep=";")
    df = clean_columns(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=15,
                    sparse_output=False,
                ),
                NOMINAL_COLS,
            ),
            ("numeric", StandardScaler(), numeric_cols),
        ]
    )


def model_search_spaces() -> dict:
    """Estimator + small hyper-parameter grid for each required classifier."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=SEED),
            {
                "clf__C": [0.05, 0.1, 0.5, 1.0, 5.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [4, 6, 8, 10, None],
                "clf__min_samples_leaf": [1, 5, 10, 20],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [5, 11, 17, 25, 35],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": np.logspace(-9, 0, 19)},
        ),
        # Platt-scaled RBF SVM: sigmoid calibration provides predict_proba for AUC
        "Support Vector Machine": (
            CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", ensemble=False),
            {
                "clf__estimator__C": [0.5, 1.0, 5.0],
                "clf__estimator__gamma": ["scale", 0.005],
            },
        ),
    }


def main() -> None:
    start = time.time()
    print(f"Python {platform.python_version()} | scikit-learn {sklearn.__version__}")

    df = load_data()
    print(f"Dataset shape after de-duplication: {df.shape}")
    print("Class distribution:\n", df[TARGET_COL].value_counts().to_string(), "\n")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    numeric_cols = [c for c in X.columns if c not in NOMINAL_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    test_df.to_csv(TEST_DATA_PATH, index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params = [], {}

    for name, (estimator, grid) in model_search_spaces().items():
        t0 = time.time()
        pipe = Pipeline(
            [("prep", build_preprocessor(numeric_cols)), ("clf", estimator)]
        )
        search = GridSearchCV(pipe, grid, scoring="f1_weighted", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        metrics, cm, _ = evaluate_model(best, X_test, y_test)
        rows.append({"ML Model Name": name, **metrics})
        best_params[name] = {
            k.replace("clf__", "").replace("estimator__", ""): (
                float(v) if isinstance(v, np.floating) else v
            )
            for k, v in search.best_params_.items()
        }
        joblib.dump(best, MODEL_DIR / MODEL_FILES[name])

        print(f"[{name}] done in {time.time() - t0:.1f}s")
        print(f"  best params : {best_params[name]}")
        print(f"  CV F1 (wtd) : {search.best_score_:.4f}")
        print("  test        : " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        print(f"  confusion matrix (rows=true {[str(c) for c in best.classes_]}):\n{cm}\n")

    results = pd.DataFrame(rows).set_index("ML Model Name")[METRIC_NAMES].round(4)
    results.to_csv(METRICS_PATH)

    metadata = {
        "dataset": "UCI Predict Students' Dropout and Academic Success (id 697)",
        "target": TARGET_COL,
        "classes": sorted(y.unique().tolist()),
        "feature_columns": X.columns.tolist(),
        "nominal_columns": NOMINAL_COLS,
        "n_rows_after_dedup": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_seed": SEED,
        "best_params": best_params,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print("=" * 80)
    print("Comparison table (held-out test set)\n")
    print(results.to_string())
    print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
