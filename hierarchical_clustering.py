import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


df = pd.read_csv("insurance.csv")
df = pd.get_dummies(df, drop_first=True)
print(df.head())
scaler = StandardScaler()
X = scaler.fit_transform(df)
methods = ['ward', 'complete', 'average']

for method in methods:
    plt.figure(figsize=(8,5))
    linked = linkage(X, method=method)
    dendrogram(linked)
    plt.title(f"Dendrogram ({method})")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()

def calculate_wcss(X, labels):
    clusters = np.unique(labels)
    wcss = 0
    for c in clusters:
        cluster_points = X[labels == c]
        center = cluster_points.mean(axis=0)
        wcss += np.sum((cluster_points - center)**2)
    return wcss

for method in methods:
    wcss_values = []
    K = range(2,7)

    for k in K:
        model = AgglomerativeClustering(n_clusters=k, linkage=method)
        labels = model.fit_predict(X)
        wcss_values.append(calculate_wcss(X, labels))

    plt.figure()
    plt.plot(K, wcss_values)
    plt.title(f"Elbow Method ({method})")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()

for method in methods:
    print(f"\nMethod: {method}")
    for k in range(2,7):
        model = AgglomerativeClustering(n_clusters=k, linkage=method)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"k = {k} -> Silhouette Score = {score:.3f}")

final_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
df['Cluster'] = final_model.fit_predict(X)
print("\nCluster Means:")
print(df.groupby("Cluster").mean())

plt.figure()
plt.scatter(df['age'], df['charges'], c=df['Cluster'])
plt.xlabel("Age")
plt.ylabel("Charges")
plt.title("Clusters Visualization")
plt.show()
