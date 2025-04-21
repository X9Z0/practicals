Here you go — a complete write-up of the **Single Layer Perceptron Learning Algorithm** implementation, including **Aim**, **Concept**, **Procedure**, and **Conclusion** for academic or project reports:

---

### 🧪 **Aim:**

To implement the **Single Layer Perceptron Learning Algorithm** using Python to simulate the behavior of a simple logical AND gate.

---

### 📚 **Concept:**

A **perceptron** is the most basic type of neural network model introduced by **Frank Rosenblatt**. It is used for **binary classification** and can solve linearly separable problems such as AND and OR logic gates.

- The perceptron has:
  - A set of **inputs**.
  - A set of **weights** (one per input).
  - A **bias** input.
  - An **activation function** (typically the step function).

The learning rule updates weights according to the **error** in prediction:

\[
w_i = w_i + \eta \cdot (t - o) \cdot x_i
\]

Where:

- \( w_i \): Weight
- \( \eta \): Learning rate
- \( t \): Target output
- \( o \): Predicted output
- \( x_i \): Input

---

### 🔬 **Procedure:**

1. **Initialize** the weights and learning rate.
2. **Prepare** the input dataset (including a bias term).
3. Use a **step activation function**:
   - Output = 1 if net input > 0 else 0.
4. For each training epoch:
   - Calculate the output using current weights.
   - Compute the **error** (target - prediction).
   - Update the weights using the perceptron rule.
5. After training, test the model on input patterns to **predict** the outputs.

---

### 🧑‍💻 **Python Code:**

```python
import numpy as np

# Input for AND gate with bias term
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Target outputs
y = np.array([0, 0, 0, 1])

# Initialize weights and learning rate
weights = np.zeros(X.shape[1])
learning_rate = 0.1
epochs = 10

# Step activation function
def step(net_input):
    return 1 if net_input > 0 else 0

# Training
for epoch in range(epochs):
    for i in range(len(X)):
        net_input = np.dot(X[i], weights)
        output = step(net_input)
        error = y[i] - output
        weights += learning_rate * error * X[i]

# Prediction
def predict(x):
    x_with_bias = np.insert(x, 0, 1)
    return step(np.dot(x_with_bias, weights))

# Testing
print("Final Weights:", weights)
print("Predictions:")
for x, target in zip(X[:, 1:], y):
    pred = predict(x)
    print(f"Input: {x}, Prediction: {pred}, Expected: {target}")
```

---

### ✅ **Conclusion:**

The Single Layer Perceptron was successfully implemented and trained to simulate an AND logic gate. It learned the correct weights through repeated exposure to the training data and was able to classify inputs accurately. This simple example demonstrates how neural networks learn patterns and make decisions based on them. However, perceptrons are limited to **linearly separable** problems and cannot solve tasks like XOR without multi-layer extensions.

---

Let me know if you want this in **PDF format**, or want to try **multi-layer perceptrons** or add **visualizations**!
