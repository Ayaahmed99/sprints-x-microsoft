import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/heart_disease_cleaned.csv")
TARGET_COL = "target"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sse = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    sse.append(km.inertia_)
plt.figure()
plt.plot(k_range, sse, marker='o')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Elbow Method For KMeans')
plt.show()

sil_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))
plt.figure()
plt.plot(list(k_range), sil_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette by k")
plt.show()

k = 2
km = KMeans(n_clusters=k, random_state=42, n_init=20)
labels_km = km.fit_predict(X_scaled)
print("KMeans ARI vs true:", adjusted_rand_score(y, labels_km))
print("KMeans silhouette:", silhouette_score(X_scaled, labels_km))

sample_idx = np.random.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
Z = linkage(X_scaled[sample_idx], method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Hierarchical Dendrogram (sample)")
plt.xlabel("Sample index or (cluster size)")
plt.show()

agg = AgglomerativeClustering(n_clusters=2)
labels_agg = agg.fit_predict(X_scaled)
print("Agglomerative ARI vs true:", adjusted_rand_score(y, labels_agg))
print("Agglomerative silhouette:", silhouette_score(X_scaled, labels_agg))
