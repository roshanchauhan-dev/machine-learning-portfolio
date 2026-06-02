import numpy as np
class ScratchLinearRegression:
    def __init__(self,learning_rate=0.01,iterations=1000):
        # self. permanently saves these settings inside the model object
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        # Counts your CSV: n_samples = total rows (houses), n_features = columns (SquareFeet, Bedrooms, Age)
        n_samples, n_features = X.shape

        self.weights=np.zeros(n_features) # Starts every feature multiplier at 0 (e.g., 0*SquareFeet, 0*Bedrooms, 0*Age)
        self.bias = 0.0

        for _ in range(self.iterations):
            # Calculate current predictions (y = wX + b)
            y_predicted = np.dot(X, self.weights) + self.bias

            # dw: Find out how much of our mistake was caused by the features (columns)
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            
            # db: Find out how much of our mistake was caused by the baseline starting price
            db = (2 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw   # Fix feature multipliers: Nudge them a tiny bit to reduce the mistake
            self.bias -= self.lr * db   # Fix starting base price: Nudge it a tiny bit to reduce the mistake

    def predict(self,X):
        return np.dot(X, self.weights)+self.bias

 