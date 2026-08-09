# Decision Tree Experiments: Classification & Regression

A comprehensive implementation and hyperparameter tuning project using **Decision Trees** for both **Classification** (Iris Dataset) and **Regression** (Boston Housing Dataset) with scikit-learn.



##  Part 1: Decision Tree Classification (Iris Dataset)

In this experiment, we explore multi-class classification using the famous **Iris Dataset** loaded via Seaborn.

### 1. Workflow & Implementation

* **Dataset:** 150 samples across 4 feature columns (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) predicting plant `species`.
* **Data Split:** 70% Training and 30% Testing (`test_size=0.3`, `random_state=42`).
* **Base Model:** `DecisionTreeClassifier(max_depth=2)` to visualize structural splits.
* **Visualization:** Rendered using `sklearn.tree.plot_tree`.

### 2. Pre-Pruning & Hyperparameter Tuning (`GridSearchCV`)

To find the optimal tree structure and prevent overfitting, we ran `GridSearchCV` with 5-fold cross-validation (`cv=5`) using the following parameter grid:

```python
params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5],
    'max_features': ['sqrt', 'log2', None]
}

```

* **Result:** Achieved a **`1.0` (100%) Accuracy Score** on the test set due to the clear separability of classes in the Iris dataset.

---

##  Part 2: Decision Tree Regression (Boston Housing Dataset)

In this experiment, we transition to a continuous regression task using the **Boston Housing Dataset** fetched directly via OpenML.

### 1. Workflow & Implementation

* **Dataset:** House price prediction (`MEDV` target column cast to float).
* **Data Split:** 70% Training and 30% Testing (`test_size=0.3`, `random_state=42`).
* **Baseline Regressor:** Trained a default `DecisionTreeRegressor()`.
* **Baseline Score (R²):** **`0.76`**



### 2. Hyperparameter Tuning with GridSearchCV

To optimize the regressor, we tested various combinations of splitting criteria, maximum depth, and feature subsets using negative mean squared error (`scoring='neg_mean_squared_error'`):

```python
parameter = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'max_features': ['auto', 'sqrt', 'log2']
}

```

* **Tuned Model Score:** **`0.606`** (reflecting how deep default trees can sometimes overfit to training distributions, showing lower test scores when constrained or tuned strictly via MSE cross-validation bounds).
