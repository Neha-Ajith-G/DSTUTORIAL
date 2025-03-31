import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster

# Load the dataset
file_path = "ds\\UNSW_NB15_testing-set.csv"
df = pd.read_csv(file_path)

# Preprocess the Data
# Selecing only numerical features 
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('id')  # Remove ID column
numerical_features.remove('label')  # Remove target label

# Extract numerical data for clustering
X = df[numerical_features]

# Normalize the data using StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Step 3: Perform Hierarchical Clustering using Complete Linkage
plt.figure(figsize=(12, 6))

# Compute the linkage matrix using 'complete' method
linkage_matrix = sch.linkage(X_scaled[:1000], method='complete')  # Using a sample of 1000 for visualization

# Plot the dendrogram
sch.dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

#Cut the dendrogram to form clusters
num_clusters = 5  # Define the number of clusters
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Display the first 20 cluster assignments
print("Cluster assignments for first 20 samples:", clusters[:20])

# A usecase for this clustering: Identify Patterns in Network Traffic
# Match the clustered sample with original dataset
df_sample = df.iloc[:1000].copy()
df_sample['Cluster'] = clusters  # Assign clusters to the sample

# Analyze how many normal vs. attack records are in each cluster
print("\nCluster distribution of normal vs. attack traffic:")
print(df_sample.groupby(['Cluster', 'label']).size())

# Analyze attack categories in each cluster
if 'attack_cat' in df.columns:
    print("\nCluster distribution of different attack categories:")
    print(df_sample.groupby(['Cluster', 'attack_cat']).size())
