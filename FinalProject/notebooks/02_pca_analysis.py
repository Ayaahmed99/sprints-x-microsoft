# 02_pca_analysis.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

df = pd.read_csv("data/heart_disease_cleaned.csv")

TARGET_COL = "num"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))

pca_full = PCA(n_components=X_scaled.shape[1], random_state=42)
X_pca_full = pca_full.fit_transform(X_scaled)
explained_ratio = pca_full.explained_variance_ratio_

cumvar = np.cumsum(explained_ratio)
plt.figure(figsize=(8,5))
plt.plot(np.arange(1, len(cumvar)+1), cumvar, marker='o')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.grid(True)
plt.title("PCA - Cumulative explained variance")
plt.axhline(0.95, color='r', linestyle='--', label='95% variance')
plt.legend()
plt.show()

n_components_95 = np.searchsorted(cumvar, 0.95) + 1
print("Components to explain 95% variance:", n_components_95)

pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca2[:,0], y=X_pca2[:,1], hue=y, palette='Set1', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D projection (colored by target)")
plt.legend(title=TARGET_COL)
plt.show()

os.makedirs("models", exist_ok=True)
joblib.dump(pca2, "models/pca_2components.pkl")
