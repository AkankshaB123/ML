# System Design Document: Startup Profit Prediction & Analytics Platform

An enterprise-grade, end-to-end Machine Learning (ML) and Statistical Modeling architecture designed to ingest financial parameters from early-stage startups, perform automated categorical encoding and feature engineering, and evaluate expected net profit using linear regression models, decision tree algorithms, analytical linear algebra formulations, and iterative gradient descent optimization.

---

## 1. Executive Summary & Architecture Overview

This document details the production microservice design and analytical pipeline for predicting financial outcomes ($Profit$) based on R&D Spend, Administration Cost, Marketing Spend, and Geographic Region (State). 

The platform supports synchronous API REST calls for real-time valuation, batch processing pipelines for venture capital portfolio evaluation, and statistical diagnostic engines (multicollinearity testing, VIF analysis, and ANOVA reporting).

```
+-----------------------------------------------------------------------------------+
|                                  INFERENCE LAYER                                  |
|                                                                                   |
|   +-----------------------+              +------------------------------------+   |
|   |   Client App / UI     |              |           REST API Gateway         |   |
|   |  (Web / Dashboard)    | <----------> |           (FastAPI / Flask)        |   |
|   +-----------------------+              +-----------------+------------------+   |
+------------------------------------------------------------|----------------------+
                                                             |
                                                             v
+-----------------------------------------------------------------------------------+
|                                PREPROCESSING ENGINE                               |
|                                                                                   |
|   +-------------------+    +-------------------+    +-------------------------+   |
|   | Null / Type Check | -> | Column Transformer| -> | Dummy Variable Dropper  |   |
|   | (Data Integrity)  |    | (One-Hot Encoding)|    | (Multicollinearity Drop)|   |
|   +-------------------+    +-------------------+    +------------+------------+   |
+------------------------------------------------------------------|----------------+
                                                                   |
                                                                   v
+-----------------------------------------------------------------------------------+
|                            FEATURE SCALING & MATRICES                             |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   | Multi-Column Feature Array X = [1, State_FL, State_NY, R&D, Admin, Mktg]   |   |
|   +---------------------------------------+-----------------------------------+   |
+-------------------------------------------|---------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------+
|                              MODEL INFERENCE ENGINES                              |
|                                                                                   |
|   +------------------------------------+    +---------------------------------+   |
|   |   Multiple Linear Regression (OLS) |    |  Decision Tree Regressor        |   |
|   |     (Primary Statistical Model)    |    |   (max_depth=5, High R^2)       |   |
|   +-----------------+------------------+    +----------------+----------------+   |
|                     |                                        |                    |
|   +-----------------+------------------+    +----------------+----------------+   |
|   | Closed-Form OLS (Pseudoinverse)    |    |  Iterative Gradient Descent     |   |
|   |   beta = (X^T X)^-1 X^T y          |    |   (Feature-Normalized Engine)   |   |
|   +-----------------+------------------+    +----------------+----------------+   |
|                     |                                        |                    |
|                     +-------------------+--------------------+                    |
|                                         |                                         |
|                                         v                                         |
|                       +----------------------------------+                        |
|                       |  Output Prediction: Profit ($)   |                        |
|                       +----------------------------------+                        |
+-----------------------------------------------------------------------------------+
```

---

## 2. Pipeline Stage Specifications

### Stage 1: Dataset Schema & Ingestion Protocol
* **Data Ingestion:** Structured CSV input containing $N = 50$ startup observations.
* **Feature Schema:**
  * `R&D Spend` (continuous float): Venture investment in R&D.
  * `Administration` (continuous float): Administrative operational expenditure.
  * `Marketing Spend` (continuous float): Marketing and customer acquisition spend.
  * `State` (categorical string): Geographical registration (`New York`, `California`, `Florida`).
  * `Profit` (continuous float - Target variable $y$).

### Stage 2: Exploratory Data Analysis & Quality Gates
* **Data Integrity Checks:** 0 null values across all features (`dataset.isnull().sum() == 0`).
* **Correlation Matrix ($r$):**
  * $\text{corr}(R\&D \text{ Spend}, \text{Profit}) = 0.9729$ (Strong linear association)
  * $\text{corr}(\text{Marketing Spend}, \text{Profit}) = 0.7478$ (Moderate positive association)
  * $\text{corr}(\text{Administration}, \text{Profit}) = 0.2007$ (Weak association)
* **Outlier Profile:** Boxplot analysis indicates minor lower-bound profit anomalies isolated exclusively in `New York`. `California` exhibits the highest dispersion (maximum net profit alongside maximum operational loss).

### Stage 3: Data Preprocessing & Vector Construction
1. **Categorical Feature Encoding:** Applied `OneHotEncoder` via `ColumnTransformer` to convert `State` into dummy indicator columns ($k = 3$ categories).
2. **Dummy Variable Trap Avoidance:** Dropped the first encoded state column ($k-1 = 2$ dummy variables remaining) to ensure non-singularity in matrix operations:
   $$X_{\text{processed}} = [\text{State\_Florida}, \text{State\_New York}, \text{R\&D Spend}, \text{Administration}, \text{Marketing Spend}]$$
3. **Train-Test Stratification:** 80/20 train-test split ($N_{\text{train}} = 40$, $N_{\text{test}} = 10$, `random_state = 0`).

---

## 3. Statistical Diagnostic & Model Benchmark Metrics

All models were evaluated using identical training ($N_{\text{train}} = 40$) and testing ($N_{\text{test}} = 10$) partitions.

### A. Performance Summary

| Model Architecture | Hyperparameters / Details | Train $R^2$ | Test $R^2$ | Test MSE |
| :--- | :--- | :--- | :--- | :--- |
| **Multiple Linear Regression (OLS)** | Scikit-Learn `LinearRegression` | **0.950** | **0.9347** | $8.35 \times 10^7$ |
| **Decision Tree Regressor** | `max_depth = 5`, `random_state = 0` | **0.9992** | **0.9756** | $3.12 \times 10^7$ |
| **Analytical OLS (Closed-Form)** | Matrix Equation: $\beta = (X^T X)^{-1} X^T y$ | N/A | **0.9347** | $8.35 \times 10^7$ |
| **Gradient Descent Regressor** | $\alpha = 0.001$, $1000$ Epochs (R&D normalized) | N/A | **0.9465** | $6.85 \times 10^7$ |

---

### B. Statistical OLS Summary (`statsmodels.api`)

#### Model Regression Output
* **Dependent Variable:** `Profit` ($y$)
* **Observations ($N$):** 40
* **Degrees of Freedom Residuals:** 34
* **$R^2$ Score:** $0.950$
* **Adjusted $R^2$ Score:** $0.943$
* **$F$-statistic:** $129.7$ ($p\text{-value} = 3.91 \times 10^{-21}$)
* **Log-Likelihood:** $-421.10$ | **AIC:** $854.2$ | **BIC:** $864.3$

#### Coefficient Diagnostics

| Parameter | Feature | Coefficient ($\beta$) | Std Error | $t$-statistic | $p$-value ($P > \|t\|$) | 95% Conf. Interval | Variance Inflation Factor (VIF) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\beta_0$ | `const` | $4.255 \times 10^4$ | $8358.54$ | $5.091$ | **0.000** | $[2.56 \times 10^4, 5.95 \times 10^4]$ | **29.10** |
| $\beta_1$ | `State_Florida` ($x_1$) | $-959.28$ | $4038.11$ | $-0.238$ | $0.814$ | $[-9165.71, 7247.14]$ | **1.31** |
| $\beta_2$ | `State_New York` ($x_2$) | $699.37$ | $3661.56$ | $0.191$ | $0.850$ | $[-6741.82, 8140.56]$ | **1.32** |
| $\beta_3$ | `R&D Spend` ($x_3$) | **0.7735** | $0.055$ | **14.025** | **0.000** | $[0.661, 0.886]$ | **2.73** |
| $\beta_4$ | `Administration` ($x_4$) | $0.0329$ | $0.066$ | $0.495$ | $0.624$ | $[-0.102, 0.168]$ | **1.24** |
| $\beta_5$ | `Marketing Spend` ($x_5$) | $0.0366$ | $0.019$ | $1.884$ | **0.068** | $[-0.003, 0.076]$ | **2.45** |

* **Condition Number:** $1.49 \times 10^6$ (Indicates structural scale discrepancies across raw currency inputs, requiring standardization for iterative optimization).

---

## 4. Mathematical Optimization Engines

### A. Closed-Form Ordinary Least Squares (Analytical Solution)
Parameter coefficients are computed directly without iteration using the Moore-Penrose pseudoinverse:

$$\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

* **Calculated Coefficients (Unscaled Numeric Inputs):**
  * $\beta_0 (\text{Intercept}) = 50,122.19$
  * $\beta_1 (\text{R\&D Spend}) = 0.8057$
  * $\beta_2 (\text{Administration}) = -0.0268$
  * $\beta_3 (\text{Marketing Spend}) = 0.0272$

---

### B. Single-Variable Normalized Gradient Descent Engine
Optimization formulation for single-feature ($R\&D\text{ Spend}$) hypothesis function $h_\theta(x) = m \cdot x + b$:

1. **Feature Normalization:**
   $$x_{\text{norm}} = \frac{x - \mu_x}{\sigma_x}$$

2. **Cost Function (Mean Squared Error):**
   $$J(m, b) = \frac{1}{N} \sum_{i=1}^{N} \left( (m \cdot x_i + b) - y_i \right)^2$$

3. **Gradient Calculations & Parameter Update Rule:**
   $$\frac{\partial J}{\partial m} = -\frac{2}{N} \sum_{i=1}^{N} x_i (y_i - y_{\text{pred}}), \quad \frac{\partial J}{\partial b} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - y_{\text{pred}})$$
   $$m \leftarrow m - \alpha \frac{\partial J}{\partial m}, \quad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

* **Converged Parameters ($\alpha = 0.001$, $1000$ Epochs):**
  * **Slope ($m$):** $33,576.61$
  * **Intercept ($b$):** $96,883.71$

---

## 5. Production Microservice Architecture & API Interface

### REST API Interface (`FastAPI`)

#### Endpoint: `POST /api/v1/predict/profit`

* **Request Body:**
```json
{
  "rd_spend": 165349.20,
  "administration": 136897.80,
  "marketing_spend": 471784.10,
  "state": "New York",
  "model_choice": "decision_tree"
}
```

* **Response Body:**
```json
{
  "status": "success",
  "data": {
    "predicted_profit": 192261.83,
    "model_used": "DecisionTreeRegressor(max_depth=5)",
    "r2_confidence": 0.9756,
    "processed_features": {
      "state_florida": 0.0,
      "state_new_york": 1.0,
      "rd_spend": 165349.20,
      "administration": 136897.80,
      "marketing_spend": 471784.10
    }
  }
}
```

---

## 6. Deployment & Execution Protocols

### Prerequisites
* Python 3.8+
* Dependencies: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `fastapi`, `uvicorn`

### Local Pipeline Execution
1. **Environment Setup:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn statsmodels fastapi uvicorn
   ```
2. **Train and Serve Pipeline:**
   ```python
   import uvicorn
   # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
   ```