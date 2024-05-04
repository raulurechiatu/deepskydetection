import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster


def classify(classification_data):
    print(classification_data)
    hierarchical_cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    labels = hierarchical_cluster.fit_predict(classification_data)
    linkage_data = linkage(classification_data, method='ward', metric='euclidean')
    print(linkage_data)
    print(len(linkage_data))
    dendrogram(linkage_data)
    plt.show()
    return labels


def classify2(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    distance_matrix = linkage(scaled_data, method='ward')

    cluster_labels = fcluster(distance_matrix, 5, criterion='distance')
    show_dendrogram_plot(distance_matrix)

    # Add cluster labels as a new column to your data
    # data.insert(len(data), cluster_labels)

    # Analyze and interpret the clusters based on your data
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data['cluster'] = cluster_labels
    # for i in range(len(data)):
    #     data[i]['cluster'] = cluster_labels[i]

    # index = str(len(data[0])-1)
    print(data)
    print(cluster_labels)
    # print(index)
    print(data.groupby('cluster').describe())
    # print(data.groupby(len(data)-1))
    show_data_points_plot(data, cluster_labels)
    return data, cluster_labels


def show_dendrogram_plot(distance_matrix):
    # Visualize the dendrogram (optional)
    plt.figure(figsize=(10, 6))
    dendrogram(distance_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()


def show_data_points_plot(data, labels):
    # --- Figure 2: Data Points ---
    colors = plt.cm.viridis(labels)
    plt.figure(figsize=(8, 5))
    plt.scatter(data.index, data.values[:, 0], c=colors, s=50, alpha=0.7)  # Adjust marker size and alpha
    plt.title("Data Points")
    plt.xlabel("Sample Index")
    # plt.ylabel("Feature 1")  # Replace with the actual feature name
    plt.show()
