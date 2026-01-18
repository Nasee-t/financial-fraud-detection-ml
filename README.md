# Online Payment Fraud Detection

This project focuses on detecting fraudulent online payment transactions using machine learning models.  
It handles **extreme class imbalance**, performs **exploratory data analysis (EDA)**, and compares multiple models with **threshold tuning** to optimize fraud detection.

---

## Problem Statement

Online payment systems are highly vulnerable to fraud.  
The challenge is to correctly identify fraudulent transactions while minimizing false positives, especially when fraud cases are extremely rare.

---

## 📊 Dataset

This project uses the **Online Payment Fraud Detection** dataset from Kaggle, which contains simulated online transaction data including both legitimate and fraudulent records. The dataset includes the following features:

| Feature | Description |
|---------|-------------|
| `step` | A unit of time (1 step = 1 hour) |
| `type` | Type of transaction (e.g., CASH_OUT, TRANSFER) |
| `amount` | Transaction amount |
| `nameOrig` | Origin account identifier |
| `oldbalanceOrg` | Origin account balance before transaction |
| `newbalanceOrig` | Origin account balance after transaction |
| `nameDest` | Destination account identifier |
| `oldbalanceDest` | Destination account balance before transaction |
| `newbalanceDest` | Destination account balance after transaction |
| `isFraud` | Target label (1 if fraudulent, 0 otherwise) |

**Key points:**
- The dataset contains around **6.3 million transactions**.
- The dataset is **highly imbalanced**, with a very low fraction of fraud cases.
- It is a **binary classification problem** for detecting fraudulent transactions.

**Download link:** [Kaggle Dataset](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection/data)

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand transaction behavior and fraud patterns.

### Key Findings

- Severe class imbalance
- Fraud occurs **only in specific transaction types** (`TRANSFER`, `CASH_OUT`)
- Transaction amount is **right-skewed** with extreme outliers
- Amount alone is not a strong fraud indicator

### EDA Includes

- Target distribution
- Fraud vs non-fraud comparison
- Transaction amount analysis
- Transaction type vs fraud analysis
- Correlation analysis (numeric features only)

📁 Notebook: [View Notebook](notebooks/fraud_detection.ipynb)

---

## Feature Engineering

The following engineered features significantly improved model performance:

### Balance Inconsistency

```python
oldbalanceOrg - newbalanceOrig - amount
```
Captures abnormal balance behavior common in fraudulent transactions.

### Categorical Encoding
One-hot encoding for transaction types

---

## Models Used

### 1. Logistic Regression (Baseline)
- Used as a **benchmark model**  
- Handles imbalance using `class_weight="balanced"`  
- Probability-based predictions  
- Extensive threshold tuning  

**Results:**
- High ROC-AUC  
- Very low precision due to imbalance  
- Useful as a baseline, **not production-ready**  

---

### 2. Random Forest
- Captures **non-linear relationships**  
- Tested with different `max_depth` values  
- Threshold tuning applied  

**Results:**
- Improved recall  
- Precision remains limited due to class imbalance  
- **Overfitting risk** at higher depths  

---

### 3. XGBoost (Best Model)
- Gradient boosting with **imbalance handling**  
- Used `scale_pos_weight` to address class imbalance  
- Threshold tuning applied  

**Results:**
- Best balance between **precision and recall**  
- Highest ROC-AUC  
- Most effective fraud detection model

---

## Threshold Tuning

Instead of relying on the default 0.5 threshold, multiple thresholds were evaluated:

```python
 [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

### Why Threshold Tuning Matters

- **Lower thresholds** = higher recall (catch more fraud)  
- **Higher thresholds** = higher precision (fewer false alarms)  
- Final threshold selection depends on business requirements.

## 📈 Evaluation Metrics

Models were evaluated using:

- Confusion Matrix  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC (threshold-independent)  

**ROC-AUC** measures how well the model separates fraud from non-fraud across all thresholds.

---

## 📊 Model Performance Summary

The table below summarizes the key metrics for each model on fraud detection (class 1). Threshold tuning was applied to optimize recall and F1-score.

| Model                  | Threshold | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | ROC-AUC  |
|------------------------|-----------|-----------------|----------------|-----------------|----------|
| Logistic Regression     | 0.3       | 0.00            | 0.98           | 0.00            | 0.966    |
| Logistic Regression     | 0.5       | 0.00            | 0.91           | 0.01            | 0.966    |
| Random Forest (max_depth=10) | 0.5  | 0.47            | 0.81           | 0.59            | 0.955    |
| Random Forest (max_depth=12) | 0.5  | 0.67            | 0.54           | 0.60            | 0.956    |
| **XGBoost**                 | 0.2       | 0.31            | 0.75           | 0.44            | 0.977    |
| XGBoost                 | 0.3       | 0.30            | 0.72           | 0.43            | 0.977    |

Due to extreme class imbalance, threshold tuning was necessary. A threshold of 0.2 was selected to maximize fraud recall while keeping false positives low. The final XGBoost model achieved a ROC-AUC of 0.977 with 75% fraud recall and 31% precision, which represents a reasonable tradeoff for operational fraud monitoring where catching fraudulent transactions is prioritized over minimizing false alarms.
> **Note:**
> - In fraud detection, recall is often more important than precision because catching fraudulent transactions is critical, even if it generates some false alarms.
> - Lower thresholds generally increase **recall** (catch more fraud cases) but reduce **precision** (more false alarms).  
> - Higher thresholds increase **precision** (fewer false positives) but may miss some fraud (lower recall).

