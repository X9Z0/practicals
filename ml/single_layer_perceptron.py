import numpy as np

# Step 1: Define input data for AND gate with bias
# Format: [bias, x1, x2]
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

# Expected output for AND logic gate
y = np.array([0, 0, 0, 1])

# Step 2: Initialize weights and learning rate
weights = np.zeros(X.shape[1])
learning_rate = 0.1
epochs = 10


# Activation function (step function)
def step(net_input):
    return 1 if net_input > 0 else 0


# Step 3: Train the perceptron
for epoch in range(epochs):
    for i in range(len(X)):
        net_input = np.dot(X[i], weights)
        output = step(net_input)
        error = y[i] - output
        weights += learning_rate * error * X[i]


# Step 4: Prediction function
def predict(x):
    x_with_bias = np.insert(x, 0, 1)  # add bias
    return step(np.dot(x_with_bias, weights))


# Step 5: Testing the model
print("Final Weights:", weights)
print("Predictions:")
for x, target in zip(X[:, 1:], y):
    pred = predict(x)
    print(f"Input: {x}, Prediction: {pred}, Expected: {target}")
