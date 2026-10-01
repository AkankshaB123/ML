# System Design: KNN Ad Click Prediction Engine

This document details the architecture, mathematical foundation, data processing pipeline, and evaluation methodology for the **Ad Click Prediction System** using K-Nearest Neighbors (KNN).

---

## 1. Algorithmic Overview & Mathematical Foundations

The K-Nearest Neighbors (KNN) algorithm is a non-parametric, instance-based supervised learning method used for both classification and regression.

### 1.1 Core Decision Logic
Given a new unlabeled feature vector $x$:
1. **Locate Neighborhood**: Identify the set $N_k(x)$ of $K$ points in the training set $D$ closest to $x$.
2. **Classification (Majority Vote)**: Assign $x$ to the class $j$ that appears most frequently among its $K$ neighbors:
   $$\hat{y} = \arg\max_j \sum_{i \in N_k(x)} I(y_i = j)$$
   where $I(\cdot)$ is an indicator function ($1$ if true, $0$ otherwise).
3. **Probability Estimation**: The probability that $x$ belongs to class $j$ is calculated as:
   $$P(Y = j \mid X = x) = \frac{1}{K} \sum_{i \in N_k(x)} I(y_i = j)$$
4. **Regression (Average)**: For continuous target variables, compute the target mean of the $K$ neighbors:
   $$\hat{y} = \frac{1}{K} \sum_{i \in N_k(x)} y_i$$

### 1.2 Distance Metrics
Distance calculation determines neighbor closeness. The system supports three primary Minkowski-based metrics:

- **Minkowski Distance (General Formula)**:
  $$D(A, B) = \left( \sum_{i=1}^{n} |a_i - b_i|^q \right)^{\frac{1}{q}}$$
- **Manhattan Distance ($q = 1$)**:
  $$D(A, B) = \sum_{i=1}^{n} |a_i - b_i|$$
- **Euclidean Distance ($q = 2$)**:
  $$D(A, B) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

---

## 2. Architecture & Data Pipeline

```
+------------------+     +-----------------------+     +--------------------------+
|  Raw Data Ingest | --> | Feature Preprocessing | --> | Min-Max Feature Scaling  |
| (web_data.csv)   |     | & Train-Test Split    |     | (Critical for KNN)       |
+------------------+     +-----------------------+     +--------------------------+
                                                                    |
                                                                    v
+------------------+     +-----------------------+     +--------------------------+
| Evaluation &     | <-- | Hyperparameter Tuning | <-- | KNN Model Fitting &      |
| Performance      |     | (GridSearchCV K=1..10)|     | Neighborhood Inference   |
+------------------+     +-----------------------+     +--------------------------+
```

### 2.1 Feature Definitions
The dataset consists of user interaction attributes used to predict `Clicked` ($y \in \{0, 1\}$):

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Time_Spent` | Continuous | Consumer time spent on site (minutes) |
| `Age` | Continuous | Customer age (years) |
| `Avg_Income` | Continuous | Average income of user's geographical area |
| `Internet_Usage` | Continuous | Average daily internet usage (minutes) |
| `Ad_Topic` | Categorical/Numerical | Encoded headline topic |
| `Country_Name` | Categorical/Numerical | Encoded consumer country |
| `City_Name` | Categorical/Numerical | Encoded consumer city |
| `Sex` | Binary | Gender indicator |
| `Time_Period` | Categorical/Numerical | Time of activity |
| `Weekday` | Discrete | Day of the week |
| `Month` | Discrete | Month of the year |

### 2.2 Feature Normalization Requirement
KNN relies heavily on distance computations. Unscaled features with large ranges (e.g., `Avg_Income`) will dominate metrics over smaller ranges (e.g., `Age`). 

**Min-Max Scaling Formula:**
$$X_{\text{scaled}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

---

## 3. Implementation Workflow

### 3.1 Dependencies
```text
pandas==1.3.5
numpy==1.21.6
matplotlib==3.5.3
seaborn==0.12.1
scikit-learn==1.0.2
scipy==1.7.3
```

### 3.2 Data Preparation & Scaling
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load data
add_data = pd.read_csv("web_data.csv")

TargetVariable = 'Clicked'
Predictors = [
    'Time_Spent', 'Age', 'Avg_Income', 'Internet_Usage', 'Ad_Topic', 
    'Country_Name', 'City_Name', 'Sex', 'Time_Period', 'Weekday', 'Month'
]

X = add_data[Predictors].values
y = add_data[TargetVariable].values

# Min-Max Normalization
PredictorScaler = MinMaxScaler()
PredictorScalerFit = PredictorScaler.fit(X)
X_scaled = PredictorScalerFit.transform(X)

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

### 3.3 Model Training & Probability Prediction
```python
from sklearn.neighbors import KNeighborsClassifier

# Initialize KNN Classifier with K=3
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(X_train, y_train)

# Generate class probability distributions
probabilities = knn.predict_proba(X_test)
predictions = knn.predict(X_test)
```

### 3.4 Hyperparameter Tuning (Optimal $K$ Search)
```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([("knn", KNeighborsClassifier())])
search_space = [{"knn__n_neighbors": list(range(1, 11))}]

classifier = GridSearchCV(pipe, search_space, cv=5, verbose=0)
classifier.fit(X_train, y_train)

best_k = classifier.best_estimator_.get_params()["knn__n_neighbors"]
print(f"Optimal K Value: {best_k}") # Optimal found: K = 9
```

---

## 4. Model Evaluation & Performance Metrics

### 4.1 Classification Performance ($K=3$)
- **Overall Weighted F1-Score**: `0.90`
- **Overall Accuracy**: `90%`

#### Classification Report Breakdown
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Not Clicked)** | 0.87 | 0.95 | 0.91 | 742 |
| **1 (Clicked)** | 0.93 | 0.83 | 0.88 | 590 |
| **Macro Average** | 0.90 | 0.89 | 0.89 | 1332 |
| **Weighted Average** | 0.90 | 0.90 | 0.90 | 1332 |

---

## 5. Architectural Considerations & Trade-offs

1. **Lazy Learning / Computation Cost**: KNN does not construct an explicit internal model during training. All distance calculations occur during inference ($O(N \cdot D)$ per prediction).
2. **Feature Scaling Sensitivity**: Distance metrics require features to be uniformly scaled (Min-Max scale proved superior to Standardization for this dataset).
3. **Lack of Native Feature Importance**: Unlike tree-based algorithms, standard KNN does not provide native feature importance scores.
4. **Memory Footprint**: High memory usage in production environments because the entire dataset ($X_{\text{train}}$) must remain in memory.