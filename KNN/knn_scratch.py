import numpy as np  
from collections import Counter
class KNN:
    def __init__(self, k=3):  #k is the number of neighbors the algorithm will look at to make a decision.
        self.k = k

    def fit(self, X, y):   
        #fit takes the training features (X) and their corresponding labels (y).
        #In KNN, there is no "math" (like calculating weights). 
        #You simply save the data into the object's memory so it can be used later to calculate distances.
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        """
        Iterates over the test dataset and predicts the class for each sample..
        
        Parameters:
        X_test (np.ndarray): A 2D array where each row is a sample to be predicted.
        
        Returns:
        np.ndarray: An array of predicted class labels for each sample.
        """
        # Map the single-sample prediction method (_predict) across all test samples
        # and package the results into a NumPy array for consistent output.
        return np.array([self._predict(x) for x in X_test])
    def _predict(self,x):

        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
