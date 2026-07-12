import numpy as np
class LogisticRegression:
    def __init__(self, learning_rate=0.01,iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        The magic S-curve! It squashes any raw linear output 
        into a probability between 0.0 and 1.0.
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(self.iterations):
            # A. Calculate the linear line: wX + b
            linear_model = np.dot(X, self.weights) + self.bias

            # B. Apply the Sigmoid function to get probabilities
            y_predicted = self._sigmoid(linear_model)

             # C. Calculate gradients (how much to nudge w and b)
            # dw = error * input, db = error
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # D. Update parameters (nudge downhill)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted_prob = self._sigmoid(linear_model)

        # Decision boundary: >= 0.5 is Class 1, else Class 0
        return [1 if i >= 0.5 else 0 for i in y_predicted_prob]
    