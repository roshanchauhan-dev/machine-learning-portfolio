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

## Project 2: Custom House Price Prediction (Built From Scratch)

This is my advanced Machine Learning project where I built the Linear Regression engine entirely from scratch using raw math, without relying on Scikit-Learn's training model.

### What I did
- Created a custom CSV dataset (`my_house_data.csv`) with 100 rows of housing data
- Developed a custom Python class (`ScratchLinearRegression`) inside `Scratch.py`
- Coded the **Gradient Descent** loops manually to update weights and bias
- Used Matrix Dot Products (`np.dot`) to calculate multi-feature predictions simultaneously
- Visualized results using a scatter plot combined with a 45-degree reference line

### Result
- Accuracy Score (R^2): 0.9965 (Achieved near-perfect optimization on clean data)
- Custom prediction points tightly hug the ideal baseline target on the graph

### Learning
- Mastered the core math running under the hood: y = wX + b
- Learned how a model reduces mistakes by nudging its feature multipliers (`dw`) and base value (`db`) downhill
- Learned how the **Learning Rate (`self.lr`)** acts as a safety step-size brake to prevent overcorrection
- Gained experience overriding default class parameters in Object-Oriented Python

### Tools Used
- Pure Python (OOP), numpy (matrix math), pandas (CSV reading), matplotlib (graphing)