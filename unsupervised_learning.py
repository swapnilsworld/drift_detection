import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


def plot_clusters(X, cluster_labels, title):
    plt.figure(figsize=(12, 6))
    k = len(set(cluster_labels))
    
    for cluster in range(k):
        plt.scatter(
            data.loc[data["Cluster"] == cluster, "ram_usage"],
            data.loc[data["Cluster"] == cluster, "cpu_utilization"],
            label=f"Cluster {cluster + 1}",
        )

    plt.title(title)
    plt.xlabel("RAM Usage")
    plt.ylabel("CPU Utilization")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def evaluate_and_plot_clustering(X, algorithm, title):
    cluster_labels = algorithm.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"{title}: Silhouette Score: {silhouette_avg:.2f}")
    plot_clusters(X, cluster_labels, title)
    return silhouette_avg

# K-Means Clustering
def process_kmeans():
    kmeans = KMeans(n_clusters=3, random_state=0)
    cluster_labels = kmeans.fit_predict(X)
    data["Cluster"] = cluster_labels
    return evaluate_and_plot_clustering(X, kmeans, "K-Means Clustering")


# Isolation Forest
def process_iso_forest():
    iso_forest = IsolationForest(contamination=0.05, random_state=0)
    predictions = iso_forest.fit_predict(X)
    data["Anomaly"] = predictions
    return evaluate_and_plot_clustering(X, iso_forest, "Isolation Forest Anomaly Detection")
    

# One-Class SVM
def process_one_class_svm():
    one_class_svm = OneClassSVM(nu=0.05)
    predictions = one_class_svm.fit_predict(X)
    data["Anomaly"] = predictions
    return evaluate_and_plot_clustering(X, one_class_svm, "One-Class SVM Anomaly Detection")

# PCA
def process_pca():
    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    plot_clusters(principal_components, data["Cluster"], "PCA of System Monitoring Data")
    silhouette_avg = silhouette_score(principal_components, data["Cluster"])
    print(f"PCA :: Silhouette Score: {silhouette_avg:.2f}")
    return silhouette_avg



# Load the data from the CSV file
data = pd.read_csv("system_monitoring_10k.csv")

# Define the features
features = data[
    [
        "ram_usage",
        "cpu_utilization",
        "io_bandwidth",
        "network_bandwidth",
    ]
]

# Split the data into training and testing sets
train_features, test_features = train_test_split(
    features, test_size=0.2, random_state=42
)

X = features

silhouette_scores = {
    "K-Means": process_kmeans(),
    "Isolation Forest": process_iso_forest(),
    "One-Class SVM": process_one_class_svm(),
    "PCA": process_pca()
}

# Plot consolidated chart with labels
plt.figure(figsize=(10, 6))
bars = plt.bar(silhouette_scores.keys(), silhouette_scores.values())
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Unsupervised Algorithms")
plt.ylim(0, 1)  # Adjust the y-axis limits as needed
plt.xticks(rotation=45)

# Add labels above each bar
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')

plt.show()
