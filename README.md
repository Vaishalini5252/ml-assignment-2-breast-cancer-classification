# Breast Cancer Classification — ML Assignment 2

## a. Problem Statement

Breast cancer diagnosis relies on correctly classifying a tumor as **malignant** or **benign** from a set of measurements derived from a digitized image of a fine needle aspirate (FNA) of a breast mass. This project builds, evaluates, and deploys five different machine learning classifiers to solve this **binary classification** problem, and provides an interactive Streamlit web application to let any user upload test data and inspect each model's performance without retraining.

## b. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository (also available built-in via `sklearn.datasets.load_breast_cancer`, which is the UCI dataset)
- **Instances:** 569 (≥ 500 required ✅)
- **Features:** 30 numeric features (≥ 12 required ✅) — mean, standard error, and "worst" values of 10 real-valued measurements per cell nucleus (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- **Target:** `target` — binary (0 = malignant, 1 = benign)
- **Class balance:** 212 malignant / 357 benign
- **Files produced:**
  - `breast_cancer_full.csv` — full dataset (569 rows × 31 columns)
  - `test_data.csv` — held-out 20% test split (114 rows × 31 columns) used by the Streamlit app

## c. GitHub Repository Link

> `<PASTE YOUR GITHUB REPO URL HERE AFTER PUSHING>`

Repository contains: `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, `train_models.py`, `model/` (5 trained classifiers + scaler + feature names + classification reports), `metrics_comparison.csv`.

## d. Models Used

All 5 models were trained on an 80/20 stratified train-test split of the same dataset, using `StandardScaler`-normalized features (fit on train only).

### Comparison Table

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression     | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree           | 0.9211   | 0.9163 | 0.9565    | 0.9167 | 0.9362 | 0.8341 |
| KNN                     | 0.9737   | 0.9884 | 0.9600    | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes             | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine  | 0.9825   | 0.9950 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |

### Observations

| ML Model Name           | Observation about model performance |
|--------------------------|--------------------------------------|
| Logistic Regression      | Best overall performer, tied with SVM. The classes are near-linearly separable after scaling, which plays directly to Logistic Regression's strength. Only 2 misclassifications out of 114 test samples, and the highest AUC (0.9954) of all 5 models — it ranks probabilities very reliably. |
| Decision Tree            | Weakest of the 5 models. A single tree (even depth-limited to 5) overfits the noisy boundary regions between classes and doesn't generalize as smoothly as margin- or probability-based models. Lowest MCC (0.8341) and lowest AUC, indicating its probability estimates are the least trustworthy among the five. |
| KNN                      | Strong performance, and achieved **perfect recall (1.0)** — it never missed a malignant case being passed as benign incorrectly in the positive class sense used here (no false negatives). However, it had 3 false positives, pulling precision down slightly below Logistic Regression/SVM. Performance is sensitive to feature scaling (critical since it's distance-based) and to the choice of k (k=7 used here). |
| Naive Bayes              | Decent but the weakest of the "probabilistic" models. The Gaussian Naive Bayes conditional-independence assumption across 30 correlated features (many of the 30 features are mean/SE/worst versions of the same 10 measurements, so they are strongly correlated) hurts it, since NB assumes features are independent given the class. This shows up as the second-lowest MCC (0.8492). |
| Support Vector Machine   | Matches Logistic Regression as the best model, with an RBF kernel. SVM finds a maximum-margin boundary in the scaled feature space, which works very well on this dataset's fairly clean class separation. Nearly identical scores to Logistic Regression (AUC marginally lower: 0.9950 vs 0.9954). |
| **Overall, Winner for your dataset?** | **Logistic Regression** (tied with **SVM**) — both reach 0.9825 accuracy and 0.9861 F1/Precision/Recall, with Logistic Regression having a very marginal edge in AUC (0.9954 vs 0.9950) and being far cheaper to train/infer. For a production deployment favoring simplicity and interpretability with no performance cost, **Logistic Regression is the recommended winner**. |

## Project Structure

```
ml-assignment-2/
│-- app.py                      # Streamlit app
│-- train_models.py             # Training + evaluation script (source of truth for all metrics)
│-- requirements.txt
│-- README.md
│-- test_data.csv               # Held-out test split (used by the Streamlit app)
│-- breast_cancer_full.csv      # Full dataset
│-- metrics_comparison.csv      # Comparison table (generated)
│-- model/
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- svm.joblib
│   │-- scaler.joblib           # StandardScaler fit on training data
│   │-- feature_names.joblib    # Ordered list of the 30 feature columns
│   │-- classification_reports.json
```

## How to Run Locally

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_models.py            # Regenerates models + metrics (optional, already included)
streamlit run app.py
```

## Streamlit App Features

1. **Dataset upload (CSV)** — upload `test_data.csv` (or any CSV with the same 30 features + `target` column) via the sidebar.
2. **Model selection dropdown** — choose any of the 5 trained classifiers.
3. **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1, MCC shown instantly for the selected model on the uploaded data.
4. **Confusion matrix + classification report** — visual heatmap and per-class precision/recall/F1 report.
5. **Optional all-model comparison** — a checkbox to view all 5 models' metrics side-by-side on the same uploaded data.

The app loads pre-trained models from `model/` and **never retrains** on uploaded data — it only performs inference and metric computation.

## Live Streamlit App Link

> `<PASTE YOUR DEPLOYED STREAMLIT CLOUD URL HERE>`

## BITS Virtual Lab Execution Screenshot

> `<INSERT SCREENSHOT HERE / reference screenshot filename in repo>`
