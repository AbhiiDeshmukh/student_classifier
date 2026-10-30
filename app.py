"""
Student Outcome Classifier - Streamlit front-end for ML Assignment 2.

Loads the five pre-trained pipelines from model/ and evaluates them on a test CSV
(uploaded by the user or the bundled test_data.csv). No retraining happens here.
"""

import io
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from evaluation import (
    METADATA_PATH,
    METRIC_NAMES,
    MODEL_DIR,
    MODEL_FILES,
    TARGET_COL,
    TEST_DATA_PATH,
    clean_columns,
    evaluate_model,
)

st.set_page_config(page_title="Student Outcome Classifier", layout="wide")


@st.cache_resource
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}


def read_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    # Accept the original semicolon-separated UCI export as well
    if df.shape[1] == 1 and ";" in df.columns[0]:
        df = pd.read_csv(io.BytesIO(raw), sep=";")
    return clean_columns(df)


def validate(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    problems = []
    if TARGET_COL not in df.columns:
        problems.append(f"Target column **`{TARGET_COL}`** is missing.")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        problems.append(
            f"{len(missing)} required feature column(s) missing: "
            + ", ".join(f"`{c}`" for c in missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )
    return problems


def plot_confusion_matrix(cm, classes, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=classes, yticklabels=classes, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_comparison(results: pd.DataFrame):
    long = results.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=long, x="Metric", y="Score", hue="Model", ax=ax, palette="viridis")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=5, frameon=False)
    ax.set_xlabel("")
    fig.tight_layout()
    return fig


models = load_models()
metadata = load_metadata()
feature_cols = metadata.get("feature_columns") or list(
    next(iter(models.values())).feature_names_in_
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Test data")
    source = st.radio(
        "Data source",
        ["Upload a CSV", "Use bundled test_data.csv"],
        help="Streamlit's free tier has limited resources, so only upload test data.",
    )
    uploaded = None
    if source == "Upload a CSV":
        uploaded = st.file_uploader("Test CSV (features + `Target`)", type=["csv"])

    st.download_button(
        "Download sample test_data.csv",
        data=TEST_DATA_PATH.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        width="stretch",
    )

    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))

    st.divider()
    st.caption(
        "Models were trained offline (`train_models.py`) on 80% of the UCI "
        "*Predict Students' Dropout and Academic Success* dataset. "
        "This app only loads and evaluates them."
    )

# ---------------------------------------------------------------- header
st.title("Student Outcome Classifier")
st.markdown(
    "Predicts whether a higher-education student will **Drop out**, stay **Enrolled**, "
    "or **Graduate**, using demographic, socio-economic and first-year academic data."
)

if source == "Upload a CSV" and uploaded is None:
    st.info(
        "Upload a test CSV from the sidebar, or switch to **Use bundled test_data.csv**. "
        f"The file must contain the {len(feature_cols)} feature columns and a `{TARGET_COL}` column."
    )
    st.stop()

raw_bytes = uploaded.getvalue() if uploaded is not None else TEST_DATA_PATH.read_bytes()
try:
    data = read_csv(raw_bytes)
except Exception as exc:  # malformed upload
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

problems = validate(data, feature_cols)
if problems:
    st.error("The uploaded file does not match the expected schema:\n\n- " + "\n- ".join(problems))
    st.stop()

data = data.dropna(subset=[TARGET_COL])
X = data[feature_cols]
y = data[TARGET_COL].astype(str)

model = models[model_name]
metrics, cm, report = evaluate_model(model, X, y)
classes = list(model.classes_)

tab_eval, tab_compare, tab_pred, tab_data = st.tabs(
    ["Model evaluation", "Compare all models", "Predictions", "Data preview"]
)

# ---------------------------------------------------------------- tab 1
with tab_eval:
    st.subheader(f"{model_name} - {len(data)} test rows")
    cols = st.columns(len(METRIC_NAMES))
    for col, metric in zip(cols, METRIC_NAMES):
        value = metrics[metric]
        col.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")
    st.caption(
        "Precision, Recall and F1 are support-weighted averages across classes; "
        "AUC is one-vs-rest weighted; MCC is the multi-class generalisation."
    )

    left, right = st.columns([1, 1.2])
    with left:
        st.pyplot(plot_confusion_matrix(cm, classes, f"Confusion matrix - {model_name}"))
    with right:
        st.markdown("**Classification report**")
        st.dataframe(report.style.format("{:.3f}"), width="stretch")

    params = metadata.get("best_params", {}).get(model_name)
    if params:
        with st.expander("Hyper-parameters selected by 5-fold GridSearchCV"):
            st.json(params)

# ---------------------------------------------------------------- tab 2
with tab_compare:
    st.subheader("All five models on the same test data")
    rows = []
    for name, mdl in models.items():
        m, _, _ = evaluate_model(mdl, X, y)
        rows.append({"Model": name, **m})
    results = pd.DataFrame(rows).set_index("Model")[METRIC_NAMES]

    st.dataframe(
        results.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"),
        width="stretch",
    )
    best = results["F1"].idxmax()
    st.success(f"Best model on this data by weighted F1: **{best}** (F1 = {results.loc[best, 'F1']:.4f})")
    st.pyplot(plot_comparison(results))

# ---------------------------------------------------------------- tab 3
with tab_pred:
    st.subheader(f"Row-level predictions - {model_name}")
    proba = pd.DataFrame(
        model.predict_proba(X), columns=[f"P({c})" for c in classes], index=X.index
    ).round(3)
    preds = pd.DataFrame({"Actual": y, "Predicted": model.predict(X)}, index=X.index)
    preds["Correct"] = preds["Actual"] == preds["Predicted"]
    out = pd.concat([preds, proba], axis=1)

    only_wrong = st.toggle("Show misclassified rows only")
    st.dataframe(out[~out["Correct"]] if only_wrong else out, width="stretch", height=400)
    st.download_button(
        "Download predictions",
        out.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------- tab 4
with tab_data:
    st.subheader("Uploaded data")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(f"{data.shape[0]} rows x {data.shape[1]} columns")
        st.dataframe(data.head(50), width="stretch", height=350)
    with c2:
        st.markdown("**Class distribution**")
        st.bar_chart(y.value_counts())
