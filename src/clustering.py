import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
# Load the dataset
df = pd.read_csv("MTA_Subway_Customer_Journey-Focused_Metrics__Beginning_2020.csv")
# Select numeric columns for clustering
numeric_cols = ["additional platform time", "additional train time", "total_apt", "total_att", 
                "over_five_mins", "over_five_mins_perc", "customer journey time performance"]
# Ensure the selected columns are numeric
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
# Aggregate by 'line' and compute mean
agg_df = df.groupby("line")[numeric_cols].mean().reset_index()
# Standardize the numerical features
scaler = StandardScaler()
X = scaler.fit_transform(agg_df.drop(columns=["line"]))
# KMeans Clustering
k = 3  # "bad", "mediocre", "good"
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
agg_df["Cluster"] = kmeans.fit_predict(X)
# Map clusters to performance labels based on average CJTP
cluster_summary = agg_df.groupby("Cluster")["customer journey time performance"].mean()
sorted_clusters = cluster_summary.sort_values().index.tolist()
label_mapping = {sorted_clusters[0]: "bad", sorted_clusters[1]: "mediocre", sorted_clusters[2]: "good"}
agg_df["Performance_Label"] = agg_df["Cluster"].map(label_mapping)
# Visualize clusters using pairplot
sns.pairplot(agg_df, hue="Performance_Label", diag_kind="kde", palette="Set2")
plt.suptitle("Train Line Clusters (Pairplot)", y=1.02)
plt.show()
# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=agg_df["Performance_Label"], palette="Set2", s=100)
for i, name in enumerate(agg_df["line"]):
    plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=8, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering Results (PCA Reduced)")
plt.tight_layout()
plt.show()
# Show results
print("\nLine Classification by Cluster:")
print(agg_df[["line", "Cluster", "Performance_Label"]])
print("\nCluster Summary Statistics:")
print(agg_df.groupby("Performance_Label")[numeric_cols].mean())


Hierarchical 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
df = pd.read_csv("MTA_Subway_Customer_Journey-Focused_Metrics__Beginning_2020.csv")
# Select relevant numeric columns
numeric_cols = ["additional platform time", "additional train time", "total_apt", "total_att", 
                "over_five_mins", "over_five_mins_perc", "customer journey time performance"]
# Convert to numeric just in case
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
# Aggregate by 'line'
agg_df = df.groupby("line")[numeric_cols].mean().reset_index()
# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(agg_df.drop(columns=["line"]))
# DENDROGRAM 
plt.figure(figsize=(10, 6))
dendro = sch.dendrogram(sch.linkage(X, method='ward'), labels=agg_df['line'].values)
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
plt.xlabel("Subway Line")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
# Apply Agglomerative Clustering 
n_clusters = 3  # Based on visual cut of the dendrogram
model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
agg_df["Cluster"] = model.fit_predict(X)
# Labeling Based on Performance 
cluster_means = agg_df.groupby("Cluster")["customer journey time performance"].mean()
sorted_clusters = cluster_means.sort_values().index.tolist()
label_map = {sorted_clusters[0]: "bad", sorted_clusters[1]: "mediocre", sorted_clusters[2]: "good"}
agg_df["Performance_Label"] = agg_df["Cluster"].map(label_map)
# PCA Visualization 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=agg_df["Performance_Label"], palette="Set2", s=100)
for i, name in enumerate(agg_df["line"]):
    plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=9, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Subway Lines (Hierarchical Clustering with PCA)")
plt.tight_layout()
plt.show()
# Results 
print("\nCluster Assignments:")
print(agg_df[["line", "Cluster", "Performance_Label"]])
print("\nCluster Summary Statistics:")
print(agg_df.groupby("Performance_Label")[numeric_cols].mean())


