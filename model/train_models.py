"""
train_models.py
----------------
Trains five classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset and persists each fitted model (as a scikit-learn Pipeline that bundles
its own StandardScaler) to disk. Also produces:

  * data/breast_cancer_full.csv  - the complete labelled dataset
  * test_data.csv                - held-out test split used by the Streamlit app
  * model/metrics.json           - evaluation metrics for every model

Run:
    python model/train_models.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "diagnosis"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")

# Human-friendly name -> (filename stem, estimator)
MODELS = {
    "Logistic Regression": (
        "logistic_regression",
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    ),
    "Decision Tree": (
        "decision_tree",
        DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    ),
    "kNN": (
        "knn",
        KNeighborsClassifier(n_neighbors=7),
    ),
    "Naive Bayes": (
        "naive_bayes",
        GaussianNB(),
    ),
    "Random Forest (Ensemble)": (
        "random_forest",
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    ),
}


def load_dataset():
    """Return the Breast Cancer dataset as a features DataFrame and target Series."""
    bunch = load_breast_cancer(as_frame=True)
    features = bunch.data.copy()
    # 0 = malignant, 1 = benign (as provided by scikit-learn)
    target = bunch.target.rename(TARGET_COLUMN)
    return features, target


def evaluate(model, X_test, y_test):
    """Compute the six required evaluation metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_test, y_proba)), 4),
        "Precision": round(float(precision_score(y_test, y_pred)), 4),
        "Recall": round(float(recall_score(y_test, y_pred)), 4),
        "F1": round(float(f1_score(y_test, y_pred)), 4),
        "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    X, y = load_dataset()

    # Persist the full labelled dataset for reference / reproducibility.
    full = X.copy()
    full[TARGET_COLUMN] = y.values
    full.to_csv(os.path.join(DATA_DIR, "breast_cancer_full.csv"), index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # Save the held-out test split at repo root (uploaded to the Streamlit app).
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

    metrics = {}
    for name, (stem, estimator) in MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, os.path.join(HERE, f"{stem}.pkl"))
        metrics[name] = evaluate(pipeline, X_test, y_test)
        print(f"Trained {name:<26} -> {metrics[name]}")

    # Persist the feature order so the app can validate uploaded data.
    metadata = {
        "target_column": TARGET_COLUMN,
        "feature_names": list(X.columns),
        "class_labels": {"0": "malignant", "1": "benign"},
        "n_test_samples": int(len(X_test)),
    }

    with open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"metrics": metrics, "metadata": metadata}, fh, indent=2)

    print("\nSaved models, test_data.csv and metrics.json successfully.")


if __name__ == "__main__":
    main()
