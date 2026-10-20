# Machine Learning Portfolio

A collection of machine learning implementations covering regression, classification, clustering, and tree-based models. Each algorithm is implemented two ways where possible: **from scratch** using only NumPy (to understand the underlying math) and using **scikit-learn** (the industry-standard approach), so the results can be compared side by side.

---

## 1. Linear Regression

**Sklearn version:** Standard pipeline — train/test split, StandardScaler, `LinearRegression`, evaluated with R².
- Result: R² = 0.71

**From-scratch version (`Scratch.py`):** A custom `ScratchLinearRegression` class built with manual gradient descent, extended to support **Ridge (L2)** and **Lasso (L1)** regularization coded from raw math (no sklearn).
- Result: R² > 0.99 on clean data, > 0.98 on noisy data
- Showed that Lasso shrinks noisy/irrelevant feature weights toward zero, while Ridge shrinks correlated features more smoothly without zeroing them out

**Files:** `Scratch.py`, `main_custom_scratch.ipynb`, `main_boston_sklearn.ipynb`, `my_house_data.csv`

---

## 2. Logistic Regression

**From-scratch version (`LogisticRegression_Scratch.py`):** Manually implemented sigmoid function and gradient descent loop, trained on a binary subset of the Iris dataset.
- Result: 100% accuracy

**Sklearn version:** Multi-class classification on the full Iris dataset with `StandardScaler` and `LogisticRegression(max_iter=5000)`.
- Result: 100% accuracy, verified with a full classification report (precision/recall/F1)

**Files:** `LogisticRegression_Scratch.py`, `main_iris_scratch.ipynb`, `main_iris_sklearn.ipynb`, `iris.csv`

---

## 3. K-Nearest Neighbors (KNN)

**From-scratch version (`knn_scratch.py`):** Implements KNN using Euclidean distance calculation and majority voting (via `collections.Counter`), with no sklearn model used for training.
- Dataset: Iris (150 samples, 4 features, 3 classes), 80/20 split, k = 3
- Result: 92.5% accuracy

**Files:** `knn_scratch.py`, `iris_knn_scratch.ipynb`, `irisdata.csv`

---

## 4. Clustering

Unsupervised learning on the `Mall_Customers.csv` dataset for customer segmentation, comparing three clustering approaches:
- **K-Means Clustering** — partitions customers into k groups by minimizing within-cluster distance
- **Hierarchical Clustering** — builds a dendrogram to find nested customer groupings
- **DBSCAN** — density-based clustering, used to detect clusters of arbitrary shape and flag outliers

**Files:** `K_means_clustering.ipynb`, `Hierarchical_clustering.ipynb`, `DBSCAN_clustering.ipynb`, `Mall_Customers.csv`

---

## 5. Decision Trees: Classification & Regression

**Classification (Iris dataset):**
- Base model: `DecisionTreeClassifier(max_depth=2)`, visualized with `plot_tree`
- Tuned using `GridSearchCV` (5-fold cross-validation) over criterion, splitter, max_depth, and max_features
- Result: 100% accuracy

**Regression (Boston Housing dataset):**
- Baseline `DecisionTreeRegressor()` — R² = 0.76
- Tuned with `GridSearchCV` using negative MSE scoring — R² = 0.606 after tuning, showing how constraining tree depth to prevent overfitting can trade off some test performance
- Includes separate notebooks for **pre-pruning** (restricting tree growth via GridSearchCV) and **post-pruning** (cost-complexity pruning after growing a full tree), plus a dedicated cross-validation notebook

**Files:** `Decison_tree_pre_prunning.ipynb`, `Decision_tree_post_prunning.ipynb`, `Decision_tree_regression_Cross_validation.ipynb`

---

## 6. Exploratory Data Analysis (EDA)

Automated data profiling using `ydata-profiling` (formerly pandas-profiling) on the Seaborn `tips` dataset, generating a full statistical and visual report (`output.html`) covering distributions, correlations, and missing values.

**Files:** `Pandas_Profiling.ipynb`, `output.html`

---

## What This Project Demonstrates

- Understanding of the math behind ML algorithms (gradient descent, sigmoid, Euclidean distance, regularization), not just calling `.fit()`
- Ability to compare a from-scratch implementation against the equivalent sklearn model
- Practical experience with hyperparameter tuning (`GridSearchCV`), cross-validation, and pruning strategies to control overfitting
- Exposure to supervised learning (regression, classification), unsupervised learning (clustering), and automated EDA

## Tech Stack
Python, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, ydata-profiling

## Author
**Roshan Chauhan**
B.E. Artificial Intelligence and Machine Learning, A. P. Shah Institute of Technology, Thane
GitHub: [roshanchauhan-dev](https://github.com/roshanchauhan-dev)
