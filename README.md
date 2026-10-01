# Portfolio Project System Design: Multi-Model ML Pipeline & Serving Architecture

A lightweight, production-ready system design showcasing a complete Machine Learning system integrating **Linear Baselines (Logistic Regression)**, **Unsupervised Clustering (K-Means & DBSCAN)**, **Tree Ensembles (Random Forest, XGBoost, LightGBM)**, and **SHAP Explainability**. Designed to demonstrate end-to-end ML engineering capabilities for portfolio presentation.

## 1. High-Level System Architecture Overview

The platform ingests raw tabular inputs, routes data through preprocessing transformers, and feeds parallel execution branches for linear benchmarking, unsupervised cluster assignment, tree ensemble inference, and real-time model explainability.

```
┌─────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│  Client App /   │ ───> │   FastAPI Gateway /     │ ───> │  Preprocessing Engine   │
│  Dashboard      │ <─── │   REST Endpoint         │ <─── │  (Scalers, Encoders)    │
└─────────────────┘      └─────────────────────────┘      └───────────┬─────────────┘
                                                                      │
                                   ┌──────────────────────────────────┼──────────────────────────────────┐
                                   │                                  │                                  │
                                   v                                  v                                  v
                        ┌─────────────────────┐            ┌─────────────────────┐            ┌─────────────────────┐
                        │ Linear Baseline     │            │ Unsupervised        │            │ Tree Ensembles      │
                        │ (Logistic Regression)│           │ (K-Means / DBSCAN)  │            │ (RF, XGBoost, LGBM) │
                        └─────────────────────┘            └─────────────────────┘            └───────────┬─────────┘
                                                                                                          │
                                                                                                          v
                                                           ┌─────────────────────┐            ┌─────────────────────┐
                                                           │ Drift Detection     │ <───────── │ SHAP Explainability │
                                                           │ (PSI / Distribution)│            │ Engine              │
                                                           └─────────────────────┘            └─────────────────────┘

```

## 2. Machine Learning Paradigms & Algorithms

### A. Linear Baseline: Logistic Regression

Serves as the lightweight, interpretable baseline to evaluate whether complex tree ensembles yield statistically significant accuracy gains.

* **Probability Formulation:**

$$
p(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

* **Preprocessing Requirements:** Features require explicit standard scaling ($x' = \frac{x - \mu}{\sigma}$) and One-Hot Encoding for categorical variables.

* **Loss Function:** Binary Cross-Entropy (BCE) with $L_2$ regularization:

$$
J(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right] + \frac{\lambda}{2} \Vert{}\mathbf{w}\Vert{}_2^2
$$

---

### B. Unsupervised Learning & Clustering Module

Discovers latent customer segments and flags anomalies without ground-truth labels.

#### 1. K-Means Clustering (Segmentation)

Partitions inputs into $K$ spherical clusters by minimizing Within-Cluster Sum of Squares (WCSS / Inertia):

$$
J_{\text{K-Means}} = \sum_{k=1}^K \sum_{\mathbf{x} \in C_k} \Vert{}\mathbf{x} - \boldsymbol{\mu}_k\Vert{}^2
$$

* **System Role:** Assigns real-time cluster labels and segment IDs to incoming feature vectors.

#### 2. DBSCAN (Density-Based Spatial Clustering)

Groups densely packed points based on neighborhood distance ($\epsilon$) and minimum points ($\text{MinPts}$):

* **System Role:** Unsupervised anomaly detection. Outlier records falling into sparse regions are flagged as noise ($\text{Cluster} = -1$).

---

### C. Tree-Based Ensembles & Optimization

Primary engine for complex non-linear classification and regression tasks on tabular data.

```
Raw Input Data
       │
       ▼
Feature Preprocessing Pipeline
       │
       ├────────────────────────────────┬────────────────────────────────┐
       ▼                                ▼                                ▼
Random Forest                    XGBoost                         LightGBM
(Parallel Bagging / OOB)         (Level-wise Boosting)           (Leaf-wise / Histogram)
       │                                │                                │
       └────────────────────────────────┴────────────────────────────────┘
                                        │
                                        ▼
                         Selected Model Artifact
                                        │
                                        ▼
                           SHAP Interpretability Pass

```

1. **Random Forest (Bagging):** Evaluates decorrelated bootstrap trees in parallel; provides Out-of-Bag (OOB) validation scores.
2. **XGBoost (Gradient Boosting):** Optimizes second-order Taylor expansion loss functions using level-wise tree growth.
3. **LightGBM (Histogram-Based Boosting):** Bins continuous features into $256$ discrete bins for fast split finding with leaf-wise growth.

---

### D. Model Interpretability Engine (TreeSHAP)

Computes exact SHAP (SHapley Additive exPlanations) values to break down individual feature contributions:

$$
\hat{y}(\mathbf{x}) = \phi_0 + \sum_{j=1}^{M} \phi_j(\mathbf{x})
$$

Where $\phi_0$ is the base expected value and $\phi_j(\mathbf{x})$ is the contribution of feature $j$.

## 3. Preprocessing Protocol Comparison

| Paradigm / Model | Feature Scaling Needed? | Categorical Encoding | Missing Value Strategy |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **Yes** (StandardScaler) | One-Hot Encoding | Impute (Mean/Median) |
| **K-Means / DBSCAN** | **Yes** (StandardScaler) | One-Hot / Binary | Impute (Mean/Median) |
| **Tree Ensembles** | **No** (Invariant to scale) | Native / Target Encoding | Native `NaN` Branch Routing |

## 4. Unified Production API Specification

### Endpoint: `POST /api/v1/predict`

#### Request Payload

```json
{
  "model_selection": "xgboost",
  "run_clustering": true,
  "return_shap": true,
  "features": {
    "account_age_months": 24,
    "monthly_charges": 79.50,
    "total_usage_gb": 310.20,
    "support_tickets_30d": 3
  }
}
```

#### Response Payload

```json
{
  "status": "success",
  "linear_baseline": {
    "logistic_regression_prob": 0.612
  },
  "clustering": {
    "cluster_id": 2,
    "cluster_name": "High-Usage At-Risk Segment",
    "is_anomaly_dbscan": false
  },
  "tree_prediction": {
    "model_used": "xgboost_v2",
    "class_label": 1,
    "class_name": "Churn Risk",
    "probability": 0.824
  },
  "explanation": {
    "base_value": 0.150,
    "shap_values": {
      "support_tickets_30d": 0.410,
      "monthly_charges": 0.180,
      "account_age_months": -0.086
    }
  },
  "latency_ms": 2.4
}
```

## 5. MLOps, Monitoring & Performance Metrics

* **Evaluation Metrics:**
  * **Supervised Classifiers:** ROC-AUC, PR-AUC, $F_1$-Score, Log Loss.
  * **Clustering Quality:** Silhouette Score, Davies-Bouldin Index.
* **Data Drift Detection:** Population Stability Index (PSI) tracks incoming feature shift vs. training baseline:

$$
\text{PSI} = \sum_{i=1}^B \left( P_i - Q_i \right) \times \ln\left(\frac{P_i}{Q_i}\right)
$$

An alert triggers automated retraining pipelines whenever $\text{PSI} > 0.2$.