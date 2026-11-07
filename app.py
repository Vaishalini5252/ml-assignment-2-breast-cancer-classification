"""
Streamlit app: Breast Cancer Classification Model Comparison

Upload a test CSV (with the 30 original features + a 'target' column),
pick one of 5 pre-trained models, and view its evaluation metrics,
confusion matrix, and classification report -- without any retraining.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load("model/scaler.joblib")
    feature_names = joblib.load("model/feature_names.joblib")
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    return scaler, feature_names, models


scaler, feature_names, models = load_artifacts()

st.title("Breast Cancer Wisconsin — Classification Model Comparison")
st.caption(
    "Upload test data, pick a trained model, and see its evaluation metrics, "
    "confusion matrix, and classification report. Models are pre-trained — "
    "no retraining happens here."
)

# ---------------------------------------------------------------------------
# Sidebar: upload + model selection
# ---------------------------------------------------------------------------
st.sidebar.header("1. Upload Test Data (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "CSV must contain the 30 breast-cancer feature columns plus a 'target' column.",
    type=["csv"],
)

st.sidebar.header("2. Select a Model")
selected_model_name = st.sidebar.selectbox("Choose a classifier:", list(MODEL_FILES.keys()))

run_all = st.sidebar.checkbox("Also compare all 5 models on this data", value=False)

if uploaded_file is None:
    st.info("Upload a test CSV from the sidebar to begin. A sample `test_data.csv` is included in the repo.")
    st.stop()

# ---------------------------------------------------------------------------
# Load + validate data
# ---------------------------------------------------------------------------
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

missing_cols = [c for c in feature_names if c not in df.columns]
if missing_cols or "target" not in df.columns:
    st.error(
        "Uploaded CSV is missing required columns.\n\n"
        f"Missing features: {missing_cols}\n"
        f"Target column present: {'target' in df.columns}"
    )
    st.stop()

X = df[feature_names]
y_true = df["target"]

st.subheader("Preview of Uploaded Data")
st.dataframe(df.head(10), use_container_width=True)
st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

X_scaled = scaler.transform(X)


def evaluate(model, X_scaled, y_true):
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["malignant", "benign"], zero_division=0)
    return metrics, cm, report


# ---------------------------------------------------------------------------
# Single-model view
# ---------------------------------------------------------------------------
st.header(f"Results: {selected_model_name}")

model = models[selected_model_name]
metrics, cm, report = evaluate(model, X_scaled, y_true)

cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["malignant", "benign"],
        yticklabels=["malignant", "benign"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col_b:
    st.subheader("Classification Report")
    st.text(report)

# ---------------------------------------------------------------------------
# Optional: compare all 5 models
# ---------------------------------------------------------------------------
if run_all:
    st.header("Comparison Across All 5 Models")
    rows = []
    for name, m in models.items():
        met, _, _ = evaluate(m, X_scaled, y_true)
        rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in met.items()}})
    comparison_df = pd.DataFrame(rows)
    st.dataframe(comparison_df, use_container_width=True)
