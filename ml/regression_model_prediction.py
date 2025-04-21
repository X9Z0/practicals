import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Sample dataset: [Area (sqft), Bedrooms, Age (years)]
data = {
    "Area": [1000, 1500, 1800, 2400, 3000],
    "Bedrooms": [2, 3, 3, 4, 4],
    "Age": [10, 5, 8, 3, 1],
    "Price": [200000, 300000, 350000, 500000, 600000],  # Target variable
}

# Step 1: Create DataFrame
df = pd.DataFrame(data)

# Step 2: Define features and target
X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 6: Make a prediction
new_house = pd.DataFrame([[2500, 4, 2]], columns=["Area", "Bedrooms", "Age"])
predicted_price = model.predict(new_house)[0]
print(f"Predicted Price for new house: ${predicted_price:,.2f}")
