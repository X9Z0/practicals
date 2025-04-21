import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Step 1: Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA (reduce to 2 dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Visualize the results
plt.figure(figsize=(8, 6))
colors = ["r", "g", "b"]
for i, target_name in enumerate(target_names):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        alpha=0.7,
        color=colors[i],
        label=target_name,
    )

plt.legend()
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of IRIS Dataset")
plt.grid(True)
plt.tight_layout()
plt.show()
