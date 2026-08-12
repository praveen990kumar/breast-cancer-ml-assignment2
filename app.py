"""
app.py
------
Interactive Streamlit application for the ML Assignment 2.

Features
  * Upload a test CSV (defaults to the bundled test_data.csv)
  * Choose any of the five trained classifiers from a dropdown
  * View the six evaluation metrics computed live on the uploaded data
  * Inspect the confusion matrix and the full classification report
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
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

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

st.set_page_config(page_title="Breast Cancer Classifier Explorer", page_icon="🔬", layout="wide")


@st.cache_resource
def load_models():
    """Load every persisted pipeline once and cache it."""
    return {name: joblib.load(os.path.join(MODEL_DIR, fname)) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_metadata():
    with open(os.path.join(MODEL_DIR, "metrics.json"), encoding="utf-8") as fh:
        return json.load(fh)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    meta = load_metadata()["metadata"]
    target_col = meta["target_column"]
    feature_names = meta["feature_names"]
    class_labels = meta["class_labels"]

    st.title("🔬 Breast Cancer Classification Explorer")
    st.markdown(
        "Compare five machine-learning classifiers on the "
        "**Breast Cancer Wisconsin (Diagnostic)** dataset. "
        "Upload your test CSV, pick a model and inspect the results."
    )

    # ---- Sidebar controls -------------------------------------------------
    st.sidebar.header("⚙️ Controls")
    uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

    default_path = os.path.join(HERE, "test_data.csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Using uploaded file.")
    elif os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.sidebar.info("No file uploaded - using bundled test_data.csv.")
    else:
        st.warning("Please upload a test CSV to continue.")
        st.stop()

    selected_model = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

    # ---- Validate the uploaded data --------------------------------------
    if target_col not in df.columns:
        st.error(
            f"The uploaded CSV must contain the target column '{target_col}'. "
            f"Columns found: {list(df.columns)}"
        )
        st.stop()

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error(f"The uploaded CSV is missing required feature columns: {missing}")
        st.stop()

    X = df[feature_names]
    y = df[target_col]

    st.subheader("📄 Test Data Preview")
    st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)

    # ---- Run the selected model ------------------------------------------
    models = load_models()
    model = models[selected_model]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, y_pred, y_proba)

    st.subheader(f"📊 Evaluation Metrics — {selected_model}")
    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.4f}")

    left, right = st.columns(2)

    with left:
        st.subheader("🧮 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        labels = [class_labels["0"], class_labels["1"]]
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.subheader("📝 Classification Report")
        report = classification_report(
            y,
            y_pred,
            target_names=[class_labels["0"], class_labels["1"]],
            output_dict=True,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    # ---- Compare every model on the uploaded data ------------------------
    st.subheader("📈 All Models Compared (on the current test data)")
    rows = []
    for name, mdl in models.items():
        preds = mdl.predict(X)
        proba = mdl.predict_proba(X)[:, 1]
        m = compute_metrics(y, preds, proba)
        rows.append({"Model": name, **{k: round(v, 4) for k, v in m.items()}})
    comparison = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        comparison.style.highlight_max(axis=0, color="#c6f6d5"),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
