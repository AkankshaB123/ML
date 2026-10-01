# System Design Document: Bank Term Deposit Subscription Prediction Engine

A high-level architecture and system design specification for predicting whether a banking client will subscribe to a term deposit (`y = 1` or `y = 0`) following a direct telemarketing campaign.

---

## 1. System Architecture Overview

The system processes client profile metrics and historical campaign outputs, applies feature selection and dummy encoding transformations, trains a **Logistic Regression** classifier, and calculates evaluation metrics for business decisioning.

```
+-------------------+     +-----------------------+     +--------------------------+
| Data Source       | --> | Preprocessing &       | --> | One-Hot Encoding &       |
| UCI Banking.csv   |     | Feature Pruning       |     | Dummy Variable Cleanup   |
+-------------------+     +-----------------------+     +--------------------------+
                                                                     |
                                                                     v
+-------------------+     +-----------------------+     +--------------------------+
| Performance &     | <-- | Model Inference &     | <-- | Train-Test Split &       |
| ROC Metrics       |     | Evaluation            |     | Logistic Regression      |
+-------------------+     +-----------------------+     +--------------------------+
```

---

## 2. Data Ingestion & Preprocessing Pipeline

### 2.1 Raw Data Ingestion
* **Source Dataset**: `Banking.csv` (UCI Machine Learning Repository).
* **Dataset Scale**: $N = 41,188$ client records, $21$ original columns.
* **Data Quality Check**: Zero missing values detected across all attributes (`data.isnull().sum() == 0`).

### 2.2 Feature Selection & Pruning
To optimize model performance and focus on client demographic and product-holding features, $14$ redundant attributes (including temporal campaign metrics like `duration`, `pdays`, `month`, `day_of_week`, and socio-economic indicators) are pruned:

$$\text{Retained Features} = \{\text{job}, \text{marital}, \text{default}, \text{housing}, \text{loan}, \text{poutcome}\}$$

### 2.3 Categorical Encoding & Dimensionality Reduction
* **One-Hot Encoding**: Applied via `pd.get_dummies()` across the $6$ retained categorical columns.
* **Column Alignment**: Dropped $5$ redundant/unknown dummy columns (`indices: [12, 16, 18, 21, 24]`) to eliminate multicollinearity.
* **Final Transformation**: Resulted in $23$ independent feature variables ($X$) and $1$ binary target ($y$).

---

## 3. Model Training & Pipeline Architecture

```
                       +---------------------------------------+
                       |    23 Encoded Categorical Features    |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |   Train-Test Split (75% / 25%)        |
                       |   X_train: 30,891 | X_test: 10,297   |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |   Logistic Regression Model           |
                       |   Random State = 0                    |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |   Inference & Threshold Evaluation    |
                       +---------------------------------------+
```

### 3.1 Data Partitioning
* **Total Instances**: $41,188$
* **Training Set ($75\%$)**: $N = 30,891$
* **Testing Set ($25\%$)**: $N = 10,297$

### 3.2 Classifier Configuration
* **Algorithm**: `LogisticRegression(random_state=0)`
* **Target Variable ($y$)**: Binary indicator ($1 =$ Subscribed, $0 =$ Not Subscribed)

---

## 4. System Evaluation & Performance Benchmark

### 4.1 Global Metric Summary

| Evaluation Metric | Training Set | Test Set |
| :--- | :--- | :--- |
| **Total Samples** | 30,891 | 10,297 |
| **Overall Accuracy** | — | **90.0%** |
| **ROC AUC Score** | — | **0.7920** |

### 4.2 Confusion Matrices

#### Training Set Confusion Matrix ($N = 30,891$)

$$
\mathbf{CM}_{\text{Train}} = \begin{bmatrix} 27048 & 344 \\ 2852 & 647 \end{bmatrix}
$$

### 4.3 Classification Report (Test Set, $N = 10,297$)

```
              precision    recall  f1-score   support

           0       0.91      0.99      0.95      9156
           1       0.68      0.20      0.31      1141

    accuracy                           0.90     10297
   macro avg       0.79      0.59      0.63     10297
weighted avg       0.88      0.90      0.88     10297
```

---

## 5. Key System Insights & Architectural Recommendations

1. **High Baseline Accuracy vs. Imbalanced Recall**: While overall test accuracy reaches **90%**, class imbalance limits the positive subscription recall to **20%** ($228$ out of $1,141$ actual subscribers identified).
2. **Class Imbalance Mitigation**: Implement **SMOTE (Synthetic Minority Over-sampling Technique)** or adjust class weights (`class_weight='balanced'`) during training to increase subscriber identification rates.
3. **Probability Threshold Customization**: Lowering the default classification decision boundary ($\tau < 0.5$) will allow campaign managers to capture more high-intent customers at the cost of slight precision reduction.