# End-to-End Customer Segmentation & Campaign Personalization System Design

## 1. System Overview & Architecture

The objective of this system is to ingest raw customer interaction and demographic data, perform automated feature engineering, execute batch unsupervised clustering (with PCA dimensionality reduction and Agglomerative Clustering), and serve cluster assignments/insights to downstream marketing automation tools.

### High-Level Architecture Diagram

```
+-------------------+      +-----------------------+      +------------------------+
|  Source Database  | ---> | Batch Data Ingestion  | ---> |   Data Cleansing &     |
| (CRM/Transactional|      | (Apache Spark/Airflow)|      | Feature Engineering    |
+-------------------+      +-----------------------+      +------------------------+
                                                                      |
                                                                      v
+-------------------+      +-----------------------+      +------------------------+
| Downstream Activation| <---| Serving Layer / API  | <---| Model Training & Batch |
| (CRM/Ad Campaign) |      | (FastAPI / PostgreSQL)|      | Clustering Pipeline    |
+-------------------+      +-----------------------+      +------------------------+
```

---

## 2. Data Pipeline & Preprocessing Architecture

### 2.1 Ingestion & Cleaning
1. **Raw Schema**: Ingestion of customer data consisting of:
   - **Demographics**: `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer`
   - **Purchasing/Behavior**: Spend across product types (`MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`), channel metrics (`NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`), and campaign engagement (`AcceptedCmp1` to `AcceptedCmp5`, `Response`, `Complain`).
2. **Data Scrubbing**:
   - Filter rows with missing values (e.g., `Income`).
   - Remove statistical outliers (e.g., `Age > 90` or `Income > $600,000`).
   - Drop non-informative constant or identifier columns (`ID`, `Dt_Customer`, `Z_CostContact`, `Z_Revenue`).

### 2.2 Feature Engineering Rules
- **Age**: Calculated dynamically as `Current_Year - Year_Birth`.
- **Spent**: Sum total of all item category spend metrics:
  $$\text{Spent} = \text{MntWines} + \text{MntFruits} + \text{MntMeatProducts} + \text{MntFishProducts} + \text{MntSweetProducts} + \text{MntGoldProds}$$
- **Living_With**: Simplified mapping from `Marital_Status` to `Partner` (`Married`, `Together`) vs. `Alone` (`Single`, `Divorced`, `Widow`, `Alone`, `Absurd`, `YOLO`).
- **Children & Family Size**:
  $$\text{Children} = \text{Kidhome} + \text{Teenhome}$$
  $$\text{Family\_Size} = \text{Living\_With\_Count} + \text{Children}$$
- **Is_Parent**: Binary flag indicator if $\text{Children} > 0$.
- **Simplified Education**: Grouped into `Undergraduate`, `Graduate`, and `Postgraduate`.

---

## 3. Modeling Pipeline (PCA + Agglomerative Clustering)

```
[ Categorical Encoding ] ---> [ Standard Scaling ] ---> [ PCA (3 Components) ] ---> [ Agglomerative Clustering (k=4) ]
```

1. **Preprocessing & Scaling**:
   - **Label Encoding**: Map categorical text variables (`Education`, `Living_With`) to integer indices.
   - **Standardization**: Standardize features using `StandardScaler` ($\mu = 0, \sigma = 1$).

2. **Dimensionality Reduction**:
   - Apply Principal Component Analysis (**PCA**) to project high-dimensional feature space down to 3 principal components ($c_1, c_2, c_3$), retaining maximum variance while eliminating feature multi-collinearity.

3. **Clustering Specification**:
   - **Optimal K Determination**: Selected $k = 4$ via the Elbow Method (Distortion Score).
   - **Algorithm**: Agglomerative Hierarchical Clustering with Euclidean distance metric and Ward linkage.

---

## 4. Cluster Persona Profiling & Business Actionability

| Cluster | Spend Profile | Income Profile | Family & Demographic Traits | Target Marketing Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | Medium / High Spend | High Income | Older parents, family size 2–4, predominantly teenagers at home. | Premium family packages, wine & high-value item loyalty rewards. |
| **Cluster 1** | Low Spend | Low Income | Predominantly non-parents, small households ($\le 2$), lower income. | Discount deals, clearance promotions, entry-level product bundling. |
| **Cluster 2** | High Spend | High Income | Younger parents, 1 young child (non-teenager), small family size ($\le 3$). | Premium convenience offerings, catalog/web promotions, gold & gourmet food. |
| **Cluster 3** | Low Spend | Low Income | Older parents, larger families (2–5 members), teenagers, low income. | Value deals (`NumDealsPurchases`), budget family essentials. |

---

## 5. Serving & System Integration Strategy

1. **Batch Execution (Airflow Scheduled)**:
   - A weekly or daily batch pipeline runs feature engineering, PCA transformation, and cluster assignment on newly added or updated customer records.
2. **Database & API Layer**:
   - Cluster labels and engineered feature metrics are stored in a PostgreSQL database / Data Warehouse.
   - A **FastAPI** service exposes REST endpoints (e.g., `GET /customer/{id}/segment`) to fetch real-time customer segments and personalized recommendation triggers.
3. **Downstream Activation**:
   - Segment outputs are synced directly to Marketing Automation platforms (e.g., HubSpot, Salesforce Marketing Cloud) to trigger tailored email campaigns based on cluster assignment.