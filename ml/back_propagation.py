from sklearn.neural_network import MLPClassifier
import numpy as np

# XOR input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# XOR output
y = np.array([0, 1, 1, 0])

# Define MLP model
model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="logistic",
    solver="sgd",
    learning_rate_init=0.1,
    max_iter=10000,
    random_state=42,
)

# Train the model
model.fit(X, y)

# Predictions
print("Predictions on XOR input:")
for input_val in X:
    prediction = model.predict([input_val])[0]
    print(f"Input: {input_val}, Predicted: {prediction}")
