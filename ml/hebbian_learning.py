import numpy as np

# Step 1: Define input patterns for AND gate (with bias)
# Format: [bias, x1, x2]
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

# Expected output for AND gate
y = np.array([0, 0, 0, 1])

# Step 2: Initialize weights and learning rate
weights = np.zeros(X.shape[1])
learning_rate = 0.1

# Step 3: Hebbian learning rule
for i in range(len(X)):
    weights += learning_rate * X[i] * y[i]

print("Final weights after Hebbian learning:", weights)


# Step 4: Prediction function
def predict(x):
    result = np.dot(weights, x)
    return 1 if result > 0 else 0  # using a threshold of 0


# Step 5: Test predictions
test_inputs = X
print("\nPredictions after training:")
for i, x in enumerate(test_inputs):
    print(f"Input: {x[1:]}, Prediction: {predict(x)}, Expected: {y[i]}")
