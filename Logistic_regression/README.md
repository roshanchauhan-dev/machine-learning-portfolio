### 1. What is the Iris Dataset?

The **Iris dataset** is a famous collection of data used for machine learning. It contains measurements for 150 flowers belonging to three different species of the Iris plant: *Setosa*, *Versicolor*, and *Virginica*. For each flower, it records four features: **sepal length**, **sepal width**, **petal length**, and **petal width**. It is considered the "Hello World" of machine learning because it is clean and perfect for testing how well a model can learn to classify items.

### 2. How Logistic Regression Works

Logistic Regression is used to categorize data (like identifying flower species). Here is the step-by-step logic:

* **Linear Scoring**: First, the model takes the four measurements and multiplies them by "weights" and adds a "bias" to get a raw score.
* **The Sigmoid Function**: Because raw scores can be any number, we pass them through the **Sigmoid function**. This acts like a filter, turning any number into a probability between 0 and 1 (0% to 100% chance).
* **Learning (Gradient Descent)**: If the model makes a mistake, it calculates the "error". It then uses **Gradient Descent** to slightly nudge its internal weights "downhill" to reduce that error. It repeats this thousands of times until it gets the answers right.

### 3. What You Did in This Project

You built two different versions of this model to compare them side-by-side:

* **The "Scratch" Version**:
* You built a `LogisticRegression` class manually using only basic math and the `numpy` library.
* You manually coded the Sigmoid function and the gradient descent loop, proving you understand exactly how the math works under the hood.


* **The "Scikit-Learn" Version**:
* You used professional industry tools to build a pipeline.
* You added features like **StandardScaler** to normalize the data, which helps the model learn faster and more accurately.


* **Evaluation**:
* You successfully trained both models and achieved **100% accuracy** on the Iris dataset.
* You created visual graphs to plot the "decision boundary," which is the exact line the model uses to separate the different species of flowers.



---

**Sources:**

* [1] Provided context: "The Iris dataset is a classic dataset used in machine learning."
* [2] Provided context: "Iris has 3 classes: Setosa, Versicolor, Virginica."
* [3] Provided context: "X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values"
* [4] Provided context: "The Sigmoid function turns huge positive numbers into 0.99 (99% chance of Yes)..."
* [5] Provided context: "Score = (Weight * Input) + Bias"
* [6] Provided context: "The magic S-curve! It squashes any raw linear output into a probability between 0.0 and 1.0."
* [7] Provided context: "It then uses that error to 'nudge' the weights slightly... repeats this process 1,000 times..."
* [8] Provided context: "Custom Implementation: Built a LogisticRegression class from scratch using numpy matrix math."
* [9] Provided context: "Achieved 100% accuracy on both the custom (binary) and library-based (multi-class) implementations."
* [10] Provided context: "Learned how libraries simplify complex tasks like... scaling tools (to clean the data)..."