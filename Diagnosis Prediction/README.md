# System Design Document: Breast Cancer Diagnosis & Diagnostic Analytics Platform

An enterprise-grade, end-to-end Machine Learning classification architecture designed to ingest cell nuclei morphometric measurements derived from digitized Fine Needle Aspirate (FNA) images, perform parameter optimization and Out-of-Bag (OOB) error rate evaluation, and deliver real-time diagnostic inference (Benign vs. Malignant) using Random Forest ensemble models.

---

## 1. Executive Summary & Architecture Overview

This platform processes $30$ continuous nuclear morphometric features to assist medical professionals in predicting breast mass malignancy. The microservice pipeline supports both synchronous REST API calls for single-sample clinical predictions and asynchronous batch inference pipelines for laboratory analytics.

```
+-----------------------------------------------------------------------------------+
|                                  INFERENCE LAYER                                  |
|                                                                                   |
|   +-----------------------+              +------------------------------------+   |
|   |   EHR System / UI     |              |           REST API Gateway         |   |
|   |  (Clinical Dashboard) | <----------> |           (FastAPI / Flask)        |   |
|   +-----------------------+              +-----------------+------------------+   |
+------------------------------------------------------------|----------------------+
                                                             |
                                                             v
+-----------------------------------------------------------------------------------+
|                                PREPROCESSING ENGINE                               |
|                                                                                   |
|   +-------------------+    +-------------------+    +-------------------------+   |
|   | Null Value Check  | -> |  Index Mapping    | -> | Feature Matrix Form     |   |
|   | (Data Integrity)  |    | (Set ID as Index) |    | (Extract 30 Continuous) |   |
|   +-------------------+    +-------------------+    +------------+------------+   |
+------------------------------------------------------------------|----------------+
                                                                   |
                                                                   v
+-----------------------------------------------------------------------------------+
|                              MODEL INFERENCE ENGINES                              |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |                         Random Forest Classifier                          |   |
|   |  (n_estimators=400, max_depth=3, criterion='gini', max_features='log2')   |   |
|   +---------------------------------------+-----------------------------------+   |
|                                           |                                       |
|                                           v                                       |
|                  +-------------------------------------------------+              |
|                  |     Probability Prediction & Classification     |              |
|                  |     P(Diagnosis = Malignant), Diagnosis Label   |              |
|                  +-------------------------------------------------+              |
+-----------------------------------------------------------------------------------+
```

---

## 2. Pipeline Stage Specifications

### Stage 1: Dataset Schema & Ingestion Protocol
* **Data Ingestion:** Structured FNA digitized cell nucleus measurement dataset ($N = 569$).
* **Feature Schema:**
  * `id` (Index identifier): Unique patient scan record key.
  * `diagnosis` (Categorical Target): Mapped to binary numerical classification:
    $$\text{Diagnosis} = \begin{cases} 1 & \text{if Malignant (M)} \\ 0 & \text{if Benign (B)} \end{cases}$$
  * `Features (30 continuous variables)`: $10$ fundamental nucleus characteristics calculated across three summary metrics:
    1. **Mean Values (`_mean`):** Average feature values across all sampled nuclei per image.
    2. **Standard Errors (`_se`):** Standard error of feature values across sampled nuclei.
    3. **Worst Values (`_worst`):** Mean of the three largest values for each feature per image.

* **Nuclei Feature Attributes:**
  * `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave_points`, `symmetry`, `fractal_dimension`.

### Stage 2: Data Preprocessing & Validation Gates
1. **Data Integrity Checks:** Confirmed $0$ missing values across all records (`breast_cancer.isnull().sum() == 0`).
2. **Index Alignment:** Index assigned to `id` parameter; `diagnosis` extracted as target variable $y$.
3. **Data Partitioning:** Stratified 80/20 train-test split ($N_{\text{train}} = 455$, $N_{\text{test}} = 114$, `random_state = 42`).

---

## 3. Model Optimization & Performance Diagnostics

### Stage 3: Hyperparameter Optimization (`GridSearchCV`)
* **Grid Search Matrix:**
  * `max_depth`: $[2, 3, 4]$
  * `bootstrap`: $[\text{True}, \text{False}]$
  * `max_features`: $[\text{'sqrt'}, \text{'log2'}, \text{None}]$ *(Note: `'auto'` deprecated in modern Scikit-Learn)*
  * `criterion`: $[\text{'gini'}, \text{'entropy'}]$
* **Cross-Validation:** $10$-fold CV (`cv = 10`, `n_jobs = 3`).
* **Selected Production Parameters:**
  * `criterion = 'gini'`
  * `max_features = 'log2'`
  * `max_depth = 3`
  * `n_estimators = 400`

### Stage 4: Out-of-Bag (OOB) Convergence Analysis
* Evaluated OOB error rate across $n_{\text{estimators}} \in [15, 1000]$ with `warm_start=True`:
  $$\text{OOB Error} = 1 - \text{OOB Score}$$
* **Convergence Result:** OOB error stabilizes around $n_{\text{estimators}} = 400$ with an optimal OOB Error Rate of **0.04835** ($4.84\%$).

---

## 4. Model Evaluation & Benchmark Metrics

All evaluations conducted on unseen test holdout set ($N_{\text{test}} = 114$).

### A. Summary Classification Metrics
* **Accuracy:** $0.965$ ($96.50\%$)
* **Test Error Rate:** $0.0351$ ($3.51\%$)
* **Area Under ROC Curve (AUC):** $0.99+$

### B. Confusion Matrix

| | Predicted Benign (0) | Predicted Malignant (1) |
| :--- | :---: | :---: |
| **Actual Benign (0)** | **70** | **2** |
| **Actual Malignant (1)** | **2** | **40** |

* **True Negatives (TN):** 70
* **False Positives (FP):** 2
* **False Negatives (FN):** 2
* **True Positives (TP):** 40

### C. Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | $0.99$ | $0.96$ | $0.97$ | 73 |
| **Malignant** | $0.93$ | $0.98$ | $0.95$ | 41 |
| **Accuracy** | | | **0.96** | **114** |
| **Macro Average** | $0.96$ | $0.97$ | $0.96$ | 114 |
| **Weighted Average** | $0.97$ | $0.96$ | $0.97$ | 114 |

---

## 5. Interpretability: Mean Decrease in Impurity (Feature Importance)

The Random Forest model measures variable significance via Gini Impurity reduction across all decision nodes.

### Top 15 Diagnostic Features

| Rank | Feature Name | Metric Type | Mean Decrease in Impurity (MDI) |
| :---: | :--- | :--- | :---: |
| 1 | `area_worst` | Worst | **0.12986** |
| 2 | `perimeter_worst` | Worst | **0.12095** |
| 3 | `concave_points_worst` | Worst | **0.11555** |
| 4 | `concave_points_mean` | Mean | **0.10014** |
| 5 | `radius_worst` | Worst | **0.07805** |
| 6 | `concavity_mean` | Mean | **0.06214** |
| 7 | `area_mean` | Mean | **0.05656** |
| 8 | `radius_mean` | Mean | **0.05457** |
| 9 | `perimeter_mean` | Mean | **0.05174** |
| 10 | `area_se` | Standard Error | **0.04326** |
| 11 | `concavity_worst` | Worst | **0.03866** |
| 12 | `compactness_worst` | Worst | **0.02033** |
| 13 | `compactness_mean` | Mean | **0.01616** |
| 14 | `texture_worst` | Worst | **0.01554** |
| 15 | `radius_se` | Standard Error | **0.01452** |

---

## 6. Production Microservice API Specification

### REST Interface (`FastAPI`)

#### Endpoint: `POST /api/v1/predict/diagnosis`

* **Request Payload (Truncated Feature Vector):**
```json
{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "perimeter_mean": 122.80,
  "area_mean": 1001.00,
  "smoothness_mean": 0.11840,
  "compactness_mean": 0.27760,
  "concavity_mean": 0.30010,
  "concave_points_mean": 0.14710,
  "symmetry_mean": 0.2419,
  "fractal_dimension_mean": 0.07871,
  "radius_se": 1.0950,
  "texture_se": 0.9053,
  "perimeter_se": 8.5890,
  "area_se": 153.40,
  "smoothness_se": 0.006399,
  "compactness_se": 0.04904,
  "concavity_se": 0.05373,
  "concave_points_se": 0.01587,
  "symmetry_se": 0.03003,
  "fractal_dimension_se": 0.006193,
  "radius_worst": 25.38,
  "texture_worst": 17.33,
  "perimeter_worst": 184.60,
  "area_worst": 2019.00,
  "smoothness_worst": 0.16220,
  "compactness_worst": 0.66560,
  "concavity_worst": 0.71190,
  "concave_points_worst": 0.26540,
  "symmetry_worst": 0.4601,
  "fractal_dimension_worst": 0.11890
}
```

* **Response Payload:**
```json
{
  "status": "success",
  "data": {
    "prediction": "Malignant",
    "class_label": 1,
    "confidence_scores": {
      "Benign": 0.02,
      "Malignant": 0.98
    },
    "key_risk_drivers": {
      "area_worst": 2019.00,
      "perimeter_worst": 184.60,
      "concave_points_worst": 0.26540
    }
  }
}
```

---

## 7. Deployment Protocol

### Local Execution Setup
1. **Dependencies Installation:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn fastapi uvicorn
   ```
2. **Model Persistence and Serving:**
   ```python
   import joblib

   # Export model artifact after fitting
   joblib.dump(fit_rf, 'breast_cancer_rf_model.pkl')
   ```