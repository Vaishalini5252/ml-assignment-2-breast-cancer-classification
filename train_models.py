"""
Train 5 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluate each with Accuracy, AUC, Precision, Recall, F1, and MCC, and save:
  - trained models (model/*.joblib)
  - fitted scaler (model/scaler.joblib)
  - full dataset (breast_cancer_full.csv)
  - held-out test split for the Streamlit app (test_data.csv)
  - metrics comparison table (metrics_comparison.csv)
"""

import json

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
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
import joblib

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

full_df = X.copy()
full_df["target"] = y
full_df.to_csv("breast_cancer_full.csv", index=False)
print(f"Full dataset shape: {full_df.shape}")

# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the RAW (unscaled) test split with target column -> used by Streamlit app
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"Test dataset shape (saved as test_data.csv): {test_df.shape}")

# ---------------------------------------------------------------------------
# 3. Scale features (fit on train only), save scaler for inference-time reuse
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(list(X.columns), "model/feature_names.joblib")

# ---------------------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

file_names = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

# ---------------------------------------------------------------------------
# 5. Train, evaluate, save
# ---------------------------------------------------------------------------
results = []
reports = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"model/{file_names[name]}")

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append(
        {
            "ML Model Name": name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4),
        }
    )

    reports[name] = {
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["malignant", "benign"]),
    }

    print(f"\n=== {name} ===")
    print(f"Accuracy={acc:.4f}  AUC={auc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  MCC={mcc:.4f}")
    print("Confusion Matrix:\n", cm)

# ---------------------------------------------------------------------------
# 6. Save comparison table + reports
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("metrics_comparison.csv", index=False)

with open("model/classification_reports.json", "w") as f:
    json.dump(reports, f, indent=2)

print("\n\nFinal Comparison Table:\n")
print(results_df.to_string(index=False))

print("\nAll 5 models, scaler, and feature_names saved under model/")
print("Saved: breast_cancer_full.csv, test_data.csv, metrics_comparison.csv")
