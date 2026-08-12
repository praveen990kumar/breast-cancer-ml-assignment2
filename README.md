# breast-cancer-ml-assignment2
# Breast Cancer Classification — ML Assignment 2

An end-to-end machine-learning project that trains **five classification models**
on the Breast Cancer Wisconsin (Diagnostic) dataset, evaluates them with six
metrics, and serves the results through an interactive **Streamlit** web app.

---

## a. Problem Statement

Diagnosing breast cancer from a fine-needle-aspirate (FNA) biopsy is a critical
binary-classification task: given cell-nucleus measurements, predict whether a
tumour is **malignant** (cancerous) or **benign** (non-cancerous).

The goal of this project is to build, evaluate and compare several supervised
classifiers on this dataset, then deploy an interactive app that lets a user
upload test data, pick a model, and inspect its evaluation metrics, confusion
matrix and classification report.

---

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository — available through
  `sklearn.datasets.load_breast_cancer` (originally UCI ID 17).
- **Task type:** Binary classification
- **Instances:** 569 (≥ 500 required ✔)
- **Features:** 30 numeric features (≥ 12 required ✔)
- **Target column (`diagnosis`):** `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign

The 30 features are computed from a digitised image of an FNA of a breast mass
and describe characteristics of the cell nuclei. They are the **mean**,
**standard error** and **worst** values of ten base measurements:
radius, texture, perimeter, area, smoothness, compactness, concavity,
concave points, symmetry and fractal dimension.

Data splitting: a stratified **75% / 25% train–test split** (`random_state=42`).
The held-out 25% test split (143 rows) is stored as `test_data.csv` and is the
file uploaded to the Streamlit app.

---

## c. GitHub Repository Link

> https://github.com/<your-username>/breast-cancer-ml-assignment2
>
> *(Replace with your actual repository URL after pushing the code.)*

**Live Streamlit App:**
> https://<your-app-name>.streamlit.app
>
> *(Replace with your actual Streamlit Community Cloud URL after deployment.)*

### Repository structure
```
project-folder/
│-- app.py                 # Streamlit web application
│-- requirements.txt       # Python dependencies
│-- README.md              # This file
│-- test_data.csv          # Held-out test data (uploaded to the app)
│-- .gitignore
│-- data/
│   └-- breast_cancer_full.csv   # Full labelled dataset
│-- model/
    │-- train_models.py    # Trains & saves all five models
    │-- metrics.json       # Saved evaluation metrics + metadata
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    └-- random_forest.pkl
```

---

## d. Models Used

Five classifiers were trained on the **same** dataset. Each model is stored as a
scikit-learn `Pipeline` that bundles a `StandardScaler` with the estimator, so
preprocessing is applied consistently at prediction time.

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour Classifier (k = 7)
4. Naive Bayes Classifier (Gaussian)
5. Random Forest Classifier (Ensemble, 200 trees)

### Comparison Table (metrics on the held-out test split)

| ML Model Name           | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression     | 0.9860   | 0.9977 | 0.9889    | 0.9889 | 0.9889 | 0.9700 |
| Decision Tree           | 0.9371   | 0.9186 | 0.9551    | 0.9444 | 0.9497 | 0.8657 |
| kNN                     | 0.9790   | 0.9923 | 0.9677    | 1.0000 | 0.9836 | 0.9555 |
| Naive Bayes             | 0.9371   | 0.9878 | 0.9355    | 0.9667 | 0.9508 | 0.8644 |
| Random Forest (Ensemble)| 0.9580   | 0.9950 | 0.9565    | 0.9778 | 0.9670 | 0.9098 |

*Metrics are computed for the positive class (`benign = 1`). Values are
reproducible with `random_state=42`.*

### Observations on Model Performance

| ML Model Name            | Observation about model performance |
|--------------------------|-------------------------------------|
| Logistic Regression      | **Best overall.** With scaled features the decision boundary is almost linearly separable, giving the highest accuracy (0.986), AUC (0.998) and MCC (0.970). Fast, stable and highly interpretable. |
| Decision Tree            | Weakest ranker — the lowest AUC (0.919) because a single depth-limited tree makes hard, axis-aligned splits and slightly overfits. Accuracy is acceptable but it is the least reliable at ordering probabilities. |
| kNN                      | Very strong after scaling; achieves perfect recall (1.000), so it never misses a benign case, and the second-highest MCC (0.956). Distance-based, so it depends heavily on the `StandardScaler` step. |
| Naive Bayes             | Good probability ranking (AUC 0.988) but its feature-independence assumption is violated (many features are correlated), which lowers accuracy and precision, tying it for the lowest accuracy (0.937). |
| Random Forest (Ensemble) | Robust all-rounder — high AUC (0.995) and strong recall (0.978) with no tuning. The ensemble smooths out the single tree's variance, but on this cleanly separable data it does not beat Logistic Regression. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it leads on Accuracy, AUC, Precision, F1 and MCC while remaining the simplest and most interpretable model. kNN is a close second and the best choice when maximising recall (catching every benign case) is the priority. |

---

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-train the models and regenerate artifacts
python model/train_models.py

# 3. Launch the Streamlit app
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Streamlit App Features

- **CSV upload** — upload your own test data (defaults to the bundled `test_data.csv`).
- **Model dropdown** — switch between all five trained classifiers.
- **Live evaluation metrics** — Accuracy, AUC, Precision, Recall, F1 and MCC computed on the uploaded data.
- **Confusion matrix** — heatmap of predictions vs. actual labels.
- **Classification report** — per-class precision/recall/F1.
- **All-models comparison table** — every model scored on the current test data with the best value in each column highlighted.

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New App**, select this repository and the `main` branch.
4. Set the main file to `app.py` and click **Deploy**.
