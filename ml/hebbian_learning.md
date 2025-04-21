Sure! Here’s a full implementation of the **Hebbian Learning algorithm** in Python, along with the **Aim**, **Concept**, **Procedure**, and **Conclusion** — perfect for documentation or report writing.

---

## 🧪 **Aim:**

To implement the **Hebbian Learning Rule** using Python for a simple neural network and demonstrate how weights are updated based on neuron co-activation.

---

## 📚 **Concept:**

**Hebbian Learning** is one of the oldest and simplest learning rules in neural networks, introduced by Donald Hebb in 1949. It’s often summarized as:

> “**Neurons that fire together, wire together.**”

In mathematical terms, the **weight update rule** is:

\[
\Delta w_i = \eta \cdot x_i \cdot y
\]

Where:

- \( \Delta w_i \): Change in weight for input \( i \)
- \( \eta \): Learning rate (a small constant)
- \( x_i \): Input
- \( y \): Output (activation of the neuron)

Hebbian learning is **unsupervised** — it does not require labeled outputs — and is mainly used for **pattern recognition** and **associative memory**.

---

## 🔬 **Procedure:**

1. **Import Libraries:**

   - Use `numpy` for vectorized computations.

2. **Initialize Parameters:**

   - Define inputs and desired outputs (in unsupervised learning, the system learns patterns).
   - Initialize weights to zero or small random numbers.
   - Set the learning rate.

3. **Apply Hebbian Learning Rule:**

   - For each training example:
     - Compute output using a simple summation or activation.
     - Update weights using Hebb’s rule.

4. **Display the Final Weights:**
   - Show how the system “learned” associations.

---

## 🧑‍💻 **Python Implementation:**

```python
import numpy as np

# Step 1: Define input patterns and outputs
# Example: 3 training samples with 2 features each
X = np.array([
    [1, -1],
    [-1, 1],
    [1, 1]
])

# Assuming output is dot product of inputs (simple linear neuron)
# In Hebbian Learning, we can define y = x for simplicity
y = X.copy()  # unsupervised Hebbian learning

# Step 2: Initialize weights and learning rate
weights = np.zeros(X.shape[1])
learning_rate = 0.1

# Step 3: Hebbian learning rule
for i in range(len(X)):
    weights += learning_rate * X[i] * y[i]

# Step 4: Show final weights
print("Final weights after Hebbian learning:", weights)
```

---

## ✅ **Conclusion:**

The Hebbian learning algorithm was implemented successfully using Python. The weights were updated based on the correlation between inputs and outputs, demonstrating the principle of neurons strengthening their connection when they are activated together. Though simple, Hebbian learning forms the basis for many complex neural learning systems, especially in the field of **unsupervised learning** and **neuroscience-inspired AI**.

---

Would you like to:

- Visualize the learning process (e.g., weight evolution)?
- Extend this to **Hebbian learning with activation functions**?
- Export this in a **PDF or Word file** for submission?

Let me know how you want to expand it!
