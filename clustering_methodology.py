import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('./data/clustering_data_normalized_encoded.csv', index_col=0)

#This part is done in order to find the best amount
#of clusters which would "optimize" the silhoutte score
#which is important but it's not the main goal when
#assessing the clustering algorithm.
silhouette = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(data)
    score = silhouette_score(data, labels)
    silhouette.append(score)

print("Silhouette score for k=2:", max(silhouette))

#Plotting silhoutte scores according to different k values for k-means
plt.figure(figsize=(8,5))
plt.plot(range(2, 10), silhouette, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different K (after PCA)')
plt.grid(True)
plt.tight_layout()
plt.show()


#Getting results for 2 cluster in 2-means

kmeans_final = KMeans(n_clusters= 2, random_state=42)
labels_km = kmeans_final.fit_predict(data)
centroids = kmeans_final.cluster_centers_

#We now want to analyze our 2 clusters and see
#what differs between them.
cluster_sizes = np.bincount(labels_km)
print("Cluster sizes:", cluster_sizes)
for cluster in range(2):
    cluster_data = data.iloc[labels_km == cluster]
    print(f"Cluster {cluster}: size={len(cluster_data)}")

    print(cluster_data[['age','sofa','icu_los', 'vent' ,'thirtyday_expire_flag','urineoutput', 'heartrate_mean']].mean())

vital_features = [
    'age', 'sofa', 'icu_los', 'vent', 'thirtyday_expire_flag',
    'urineoutput', 'heartrate_mean', 'lactate_mean', 'bun_mean',
    'wbc_mean', 'glucose_mean',
    'sysbp_mean', 'diasbp_mean', 'meanbp_mean', 'resprate_mean', 'tempc_mean', 'spo2_mean'
]

data['cluster'] = labels_km
cluster_means = data.groupby('cluster')[vital_features].mean().T
cluster_means.columns = ['Cluster 0', 'Cluster 1']

#We plot the graph of the comparison of different features between the two clusters
plt.figure(figsize=(14, 8))
cluster_means.plot(kind='bar', figsize=(16, 8), width=0.8)
plt.title("Comparison of Vital Signs and Lab Values Between Clusters (KMeans, k=2)")
plt.ylabel("Normalized Mean Value")
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.legend()
plt.show()
#This is a generalization of the comaprison between the clusters


#The 2 clusters are built by  116-dimensional vector
#however we do want to gain an understanding
#about the clusters and so we graph them
#but in order to graph them we reduce the dimensionality
#this is done using a dimensionality reduction algorithm
#known as PCA this isn't really supposed to represent
#accurately the clusters rather give some visual intuition.
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(data.drop(columns= ['cluster']))


plt.figure(figsize=(6,5))
for cluster in range(2):
    pts = X_2d[labels_km == cluster]
    plt.scatter(pts[:,0], pts[:,1], s=20, alpha=0.6, label=f'Cluster {cluster}')

centroids_2d = pca.transform(centroids)
plt.scatter(centroids_2d[:,0], centroids_2d[:,1], c='red', marker='X', s=100, edgecolors='black', linewidths=1.5, label='Centroids')
plt.title(f'KMeans Clusters (k={2}) in PCA Space')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.legend()
#plt.show()

#DBSCAN Clustering methodology code we simply choose the DBSCAN which provides the most overall look over the data
eps_values = [2.0, 3.0, 4.0, 5.0, 6.0]
min_samples_values = [3, 5, 10]
for eps in eps_values:
    for min_pts in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_pts)
        labels_db = dbscan.fit_predict(data)
        n_clusters = len(set(labels_db) - {-1})
        n_noise = np.sum(labels_db == -1)
        print(f"eps={eps}, min_samples={min_pts}: clusters={n_clusters}, noise={n_noise}")


#DBSCAN picked, other hyper parameters either had one cluster or too many noise points.
dbscan_final = DBSCAN(eps=5.0, min_samples=3)
labels_db = dbscan_final.fit_predict(data)
n_clusters = len(set(labels_db) - {-1})
n_noise = np.sum(labels_db == -1)
print(f"Final DBSCAN: clusters={n_clusters}, noise_points={n_noise}")

if n_clusters > 0:
    sil_db = silhouette_score(data, labels_db)
    print("DBSCAN silhouette (including noise as its own cluster):", sil_db)


#Plotting the clusters of the DBSCAN for similar understanding as the K-means
plt.figure(figsize=(6,5))
unique_labels = sorted(set(labels_db))
for lab in unique_labels:
    pts = X_2d[labels_db == lab]
    if lab == -1:
        plt.scatter(pts[:,0], pts[:,1], c='gray', marker='x', s=30, alpha=0.7, label='Noise')
    else:
        plt.scatter(pts[:,0], pts[:,1], s=20, alpha=0.7, label=f'Cluster {lab}')
plt.title('DBSCAN Clusters in PCA Space')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.legend()
plt.show()

#We get out of the DBSCAN output that the results aren't as good and as representative of the reality as the clusters in k-means out of this
#We will emphasize our conclusions on k-means
#In this code we use 3 different unsupervised learning algorithms: PCA, K-means and DBSCAN.
