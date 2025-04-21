Here’s a complete implementation of the **Error Backpropagation Perceptron Training Algorithm** (also known as the **Backpropagation Algorithm**) for a **Multi-Layer Perceptron (MLP)** using Python and `scikit-learn`. It includes:

- Python Code
- Aim
- Concept
- Procedure
- Conclusion

---

### 🧪 **Aim:**

To implement the **Error Backpropagation Training Algorithm** for a Multi-Layer Perceptron (MLP) using Python.

---

### 📚 **Concept:**

Backpropagation is a **supervised learning algorithm** used for training multi-layer neural networks. It minimizes the **mean squared error** between predicted and actual output by **propagating the error backward** from the output layer to the input layer.

Key concepts:

- **Forward pass**: Calculate the output.
- **Loss calculation**: Compute the error.
- **Backward pass**: Update weights using gradients from the error.

The algorithm uses the **gradient descent** method for optimization.

---

### 🔬 **Procedure:**

1. **Import** necessary libraries.
2. **Create** a dataset (e.g., XOR).
3. **Define** the MLP model using `MLPClassifier` from `scikit-learn`.
4. **Train** the model using the dataset.
5. **Evaluate** predictions after training.

---

### 🧑‍💻 **Python Code:**

```python
from sklearn.neural_network import MLPClassifier
import numpy as np

# XOR input data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# XOR output
y = np.array([0, 1, 1, 0])

# Define MLP model
model = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=10000, random_state=42)

# Train the model
model.fit(X, y)

# Predictions
print("Predictions on XOR input:")
for input_val in X:
    prediction = model.predict([input_val])[0]
    print(f"Input: {input_val}, Predicted: {prediction}")
```

---

### ✅ **Conclusion:**

The Error Backpropagation Algorithm was successfully implemented using a Multi-Layer Perceptron. The model learned to classify the XOR logic gate—a task that single-layer perceptrons cannot solve. The backpropagation algorithm adjusted the weights layer by layer to minimize the error and achieve accurate predictions. This forms the foundation of modern deep learning.

---

Let me know if you'd like to:

- Implement this **from scratch (no libraries)**
- Try on other datasets like **AND/OR**
- Visualize the **loss curve** or **decision boundary**
- Export as **PDF or Jupyter Notebook**
