from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from scipy.stats import mode
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
species_names = iris.target_names

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Map cluster numbers to species using majority voting
df = pd.DataFrame(X, columns=iris.feature_names)
df['True_Label'] = y
df['Cluster'] = clusters

# Create a mapping from cluster to species
label_map = {}
for cluster in np.unique(clusters):
    true_labels = df[df['Cluster'] == cluster]['True_Label']
    most_common = mode(true_labels, keepdims=False).mode
    label_map[cluster] = species_names[most_common]

# Add mapped species names
df['Cluster_Species'] = df['Cluster'].map(label_map)
df['Actual_Species'] = [species_names[i] for i in y]

# Plot clusters with centroid and species names
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df.iloc[:, 0], y=df.iloc[:, 1],
    hue=df['Cluster_Species'], palette='Set2', s=100, edgecolor='k'
)

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    c='black', s=200, marker='X', label='Centroids'
)

plt.title("K-Means Clustering on Iris Dataset (Labeled by Species)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend(title='Clustered Species')
plt.grid(True)
plt.show()

# Show sample comparison
print(df[['Actual_Species', 'Cluster_Species']].head(10))
