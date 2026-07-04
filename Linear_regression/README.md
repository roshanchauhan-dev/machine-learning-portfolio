# 🏡 Machine Learning Portfolio: Linear Regression

This project showcases two different approaches to predicting house prices: using a pre-built machine learning library (Scikit-Learn) and building the mathematical training engine entirely from scratch.

---

## Project 1: House Price Prediction (Linear Regression with Sklearn)

This is my first Machine Learning project using Linear Regression.

### What I did
- Loaded dataset using sklearn
- Split data into training and testing
- Applied StandardScaler
- Trained Linear Regression model
- Predicted house prices
- Evaluated model using R² score
- Visualized results using graph

### Result
- Accuracy Score: 0.71
- Predictions are close to actual values

### Learning
- Understood how ML model works
- Learned train-test split
- Learned feature scaling
- Learned model evaluation

### Tools Used
- Python, pandas, numpy, scikit-learn, matplotlib

---

## Project 2: Custom Linear Regression Engine with L1 & L2 Regularization (Built From Scratch)

This is my advanced Machine Learning project where I built the Linear Regression engine entirely from scratch using raw math, without relying on Scikit-Learn's training model. I also expanded the engine to include Ridge (L2) and Lasso (L1) regularizations.

### What I did
- Created a custom CSV dataset (my_house_data.csv) with housing data.
- Developed a custom Python class (`ScratchLinearRegression`) inside `Scratch.py`
- Coded the **Gradient Descent** loops manually to update weights and bias
- Implemented L1 (Lasso) and L2 (Ridge) mathematical penalties from scratch to handle the noisy dataset.
- Visualized the performance of Normal, Ridge, and Lasso models side-by-side using Matplotlib subplots.

### Result
- Accuracy Score (R²): Achieved >0.99 on clean data and maintained >0.98 on noisy data.
- Feature Selection: Proved mathematically that the Lasso penalty combats noisy data by violently shrinking the weights of "garbage" columns down to near zero.
- Weight Shrinkage: Proved that the Ridge penalty identifies duplicate/competing features and safely shrinks their weights so they share the predictive workload evenly.

### Learning
- Mastered the core math running under the hood: y = wX + b
- Learned how a model reduces mistakes by nudging its feature multipliers (dw) and base value (db) downhill using calculus.
- Deeply understood the behavioral difference between L1 (flat penalty) and L2 (proportional penalty).
- Discovered the mathematical "Zero-Bounce" limitation of manual Gradient Descent, learning why professional libraries (like Scikit-Learn) use advanced calculus tricks like Soft Thresholding to force weights to exactly 0.00.
- Gained practical experience in Hyperparameter Tuning by adjusting the penalty strength (alpha).

### Tools Used
- Pure Python (OOP), numpy (matrix math), pandas (CSV reading), matplotlib (graphing)