import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

data = pd.read_csv('dataIris.csv', delimiter=';')

data.head()

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

k = 3  

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.figure(figsize=(10, 5))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolor='k')

plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title("K-Means Clustering do dataset das Iris")
plt.xlabel("Comprimento da sepála")
plt.ylabel("Largura da sepála")
plt.legend()
plt.show()

k, labels

