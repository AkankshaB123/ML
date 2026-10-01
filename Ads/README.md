# System Design Document: Social Network Ads Purchase Prediction Engine

A high-level architecture and system design specification for predicting customer ad purchase intent (`Purchased = 1` vs `Purchased = 0`) based on demographic and financial signals.

## 1. System Architecture Overview

The **Social Network Ads Prediction Engine** processes customer profile metrics, standardizes feature scales, applies a binary **Logistic Regression** decision boundary, and outputs conversion probabilities to drive targeted ad delivery.

```
+---------------------+     +-------------------------+     +--------------------------+
| Data Source         | --> | Feature Ingestion &     | --> | Feature Scaling          |
| Social_Network_Ads  |     | Selection (Age, Salary) |     | (StandardScaler Z-Score) |
+---------------------+     +-------------------------+     +--------------------------+
                                                                         |
                                                                         v
+---------------------+     +-------------------------+     +--------------------------+
| Performance &       | <-- | Model Inference &       | <-- | Train-Test Split &       |
| Confusion Matrix    |     | Decision Boundary       |     | Logistic Regression      |
+---------------------+     +-------------------------+     +--------------------------+
```

---

## 2. Data Ingestion & Preprocessing Pipeline

### 2.1 Raw Data Ingestion

* **Source Dataset**: `Social_Network_Ads.csv`
* **Dataset Scale**: $N = 400$ total records
* **Target Class Distribution**:
  * `Not Purchased (0)`: 257 records ($\sim 64.25\%$)
  * `Purchased (1)`: 143 records ($\sim 35.75\%$)

### 2.2 Feature Selection & Pruning

Non-predictive identifier fields (`User ID`, `Gender`) are bypassed to optimize memory footprint and model generalization:

* **Selected Input Features ($X$)**:
  1. `Age` (Numeric, years)
  2. `EstimatedSalary` (Numeric, annual USD)
* **Target Variable ($y$)**: `Purchased` (Binary: $1 = \text{Purchased}, 0 = \text{Not Purchased}$)

### 2.3 Feature Standardization Pipeline

Because `Age` (range $\sim 18-60$) and `EstimatedSalary` (range $\sim \$15,000-\$150,000$) operate on vastly different numerical scales, Z-score feature standardization is mandatory to prevent feature dominance during Logistic Regression gradient optimization:

$$z = \frac{x - \mu}{\sigma}$$

* **Scaler Transformation**: `StandardScaler()` fitted on $X_{train}$ and applied to both $X_{train}$ and $X_{test}$.

---

## 3. Top Causal Factors (Feature Importance & Odds Ratios)

In Logistic Regression, prediction outcomes are governed by the log-odds formula:

$$\operatorname{logit}(P) = \beta_0 + \beta_1 \cdot z_{\text{Age}} + \beta_2 \cdot z_{\text{EstimatedSalary}}$$

By examining the learned model coefficients ($\beta$) after feature scaling, we identify the primary causal drivers of ad conversions:

| Ranking | Input Factor ($X_i$) | Impact Direction | System Interpretation / Odds Impact |
| :---: | :--- | :--- | :--- |
| **#1** | **`Age`** ($\beta_1 > 0$) | **Strong Positive** | **Primary Conversion Driver.** Older demographics show exponentially higher likelihood of purchasing advertised products. A 1 standard deviation increase in age produces the largest increase in purchasing odds. |
| **#2** | **`EstimatedSalary`** ($\beta_2 > 0$) | **Moderate Positive** | **Secondary Conversion Driver.** Higher disposable income strongly correlates with conversion, enabling targeted reach for high-ticket ads when combined with age constraints. |

### Causal Inference Summary
* **High Risk / Non-Buyers**: Young users ($<35$) with low-to-medium estimated salaries ($\le \$70,000$).
* **High Conversion Potential**: Users aged $\ge 40$ or users with premium salary tiers ($\ge \$100,000$) regardless of minor age variations.

---

## 4. Model Training & Pipeline Architecture

```
                       +---------------------------------------+
                       | Input Features: [Age, EstimatedSalary]|
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       | Train-Test Split (75% / 25%)          |
                       | X_train: 300     | X_test: 100        |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       | Standard Scaling (z-score fit)        |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       | Logistic Regression Classifier        |
                       | (random_state = 0)                    |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       | Inference & Decision Boundary Mapping |
                       +---------------------------------------+
```

---

## 5. System Evaluation & Performance Metrics

### 5.1 Test Set Classification Summary ($N = 100$)

| Metric | System Value | Formula / Description |
| :--- | :--- | :--- |
| **Accuracy** | **89.00%** | $\frac{TP + TN}{TP + TN + FP + FN} = \frac{24 + 65}{100}$ |
| **Precision** | **88.89%** | $\frac{TP}{TP + FP} = \frac{24}{24 + 3}$ (High confidence in ad clicks) |
| **Recall (Sensitivity)** | **75.00%** | $\frac{TP}{TP + FN} = \frac{24}{24 + 8}$ (Captures $75\%$ of all potential buyers) |
| **F1-Score** | **81.36%** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ (Harmonic balance) |

### 5.2 Confusion Matrix Analysis ($N_{\text{test}} = 100$)

$$\mathbf{CM} = \begin{bmatrix} TN = 65 & FP = 3 \\ FN = 8 & TP = 24 \end{bmatrix}$$

* **True Negatives ($TN = 65$)**: Correctly identified non-purchasers (saved ad spend).
* **False Positives ($FP = 3$)**: Non-purchasers incorrectly targeted (minor waste of ad credit).
* **False Negatives ($FN = 8$)**: Missed potential buyers (opportunity cost).
* **True Positives ($TP = 24$)**: Correctly targeted high-converting users.

---

## 6. Operational Recommendations for Ad Targeting Systems

1. **Threshold Adjustment**: To capture the $8$ missed conversion opportunities ($FN$), lower the decision probability threshold from $\tau = 0.50$ to $\tau = 0.35$ during campaign delivery.
2. **Real-time Feature Scaling**: Store the fitted `StandardScaler` mean ($\mu$) and variance ($\sigma^2$) parameters as microservice artifacts to process incoming single-user prediction requests instantly.