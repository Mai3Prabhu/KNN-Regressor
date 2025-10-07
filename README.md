# 📈 K-Nearest Neighbors (KNN) Regressor Project

> A simple machine learning project that implements a **K-Nearest Neighbors (KNN) Regressor** using scikit-learn to predict continuous values on a synthetic dataset, achieving strong performance with an R² score of **0.92**.

---

## 🚀 Overview

This project demonstrates how to use **KNN Regression** for predicting continuous outcomes based on input features.  
A synthetic dataset is generated using `make_regression`, followed by model training, testing, and performance evaluation using standard regression metrics.

---

## 🧠 Objectives

- Generate a regression dataset using `make_regression`
- Train a **KNN Regressor** model
- Evaluate model performance using:
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
- Interpret model accuracy and residual error

---

## ⚙️ Tech Stack

| Category | Libraries Used |
|-----------|----------------|
| Core | Python 3.11 |
| Data Handling | `numpy`, `pandas` |
| Machine Learning | `scikit-learn` |
| Visualization (Optional) | `matplotlib`, `seaborn` |

---

## 🧩 Project Workflow

### 1️⃣ Data Generation
Synthetic dataset created with 1000 samples and 2 features.

```python
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1000,
    n_features=2,
    noise=10,
    random_state=42
)
```
### 2️⃣ Train-Test Split

Split dataset into 67% training and 33% testing for unbiased evaluation.

### 3️⃣ Model Training

Trained a KNeighborsRegressor with 6 neighbors using the ‘auto’ algorithm.

### 4️⃣ Model Evaluation

Predicted test set values and evaluated model using standard regression metrics.

✅ Interpretation:
The model explains ~92% of the variance in the target variable, showing excellent predictive performance for a simple KNN setup.

## 🧮 Key Insights

- KNN Regressor works effectively for small to medium datasets with clear feature relationships.

- R² score of 0.92 indicates high accuracy.

- Model performance may improve further with:

   - Feature scaling (e.g., StandardScaler)

   - Tuning n_neighbors using GridSearchCV

   - Visualizing residuals for error analysis
 
## 🧰 How to Run

**1. Clone the Repository**
```
git clone <your-repo-link>
cd knn-regressor
```

**2. Install Dependencies**
```
pip install -r requirements.txt
```

**3. Run the Script**
```
python knn_regressor.py
```
