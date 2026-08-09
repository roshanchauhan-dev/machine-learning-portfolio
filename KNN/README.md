# K-Nearest Neighbors (KNN) Classifier from Scratch

A lightweight, modular implementation of the **K-Nearest Neighbors (KNN)** machine learning algorithm built completely from scratch using **Python** and **NumPy**, evaluated on the classic **Iris dataset**.

## How the Custom KNN Works

The algorithm follows the non-parametric lazy-learning approach of KNN:

1. **Fit (`fit`)**: Stores the training features (`X_train`) and labels (`y_train`) in memory without executing heavy mathematical training steps.
2. **Predict (`predict`)**: Iterates through each sample in the test set (`X_test`) and passes it to the internal prediction function.
3. **Distance Calculation (`_predict`)**: Computes the **Euclidean Distance** between a test point and all training samples:

$$\text{Distance} = \sqrt{\sum (x - x_{\text{train}})^2}$$


4. **Neighbor Selection**: Sorts the distances, isolates the indices of the top $k$ nearest neighbors, and retrieves their class labels.
5. **Majority Vote**: Uses `collections.Counter` to find the most common class label among the neighbors and outputs it as the final prediction.

## Results & Performance

* **Dataset:** Iris Dataset (150 samples, 4 features, 3 target classes)
* **Train/Test Split:** 80% Training, 20% Testing (`random_state=42`)
* **Preprocessing:** Standard Feature Scaling (`StandardScaler`) applied to ensure distance metrics are weighed equally.
* **Hyperparameters:** $k = 3$
* **Final Model Accuracy:** **`92.50%`**