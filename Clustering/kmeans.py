import numpy as np
from sklearn import datasets
from sklearn import decomposition
from sklearn import cluster
import random
import ml_helpers

class KMeans():
    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations
        self.kmeans_centroids = []

    # Initialize the centroids as random samples
    def _init_random_centroids(self, data):
        n_samples, n_features = np.shape(data)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = data[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # Return the index of the closest centroid to the sample
    def _closest_centroid(self, sample, centroids):
        closest_i = None
        closest_distance = float("inf")
        for i, centroid in enumerate(centroids):
            distance = ml_helpers.euclidean_distance(sample, centroid)
            if distance < closest_distance:
                closest_i = i
                closest_distance = distance
        return closest_i

    # Assign the samples to the closest centroids to create clusters
    def _create_clusters(self, centroids, data):
        n_samples = np.shape(data)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(data):		
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # Calculate new centroids as the means of the samples in each cluster
    def _calculate_centroids(self, clusters, data):
        n_features = np.shape(data)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(data[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # Classify samples as the index of their clusters
    def _get_cluster_labels(self, clusters, data):
        # One prediction for each sample
        y_pred = np.zeros(np.shape(data)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # Do K-Means clustering and return the centroids of the clusters
    def fit(self, data):
        # Initialize centroids
        centroids = self._init_random_centroids(data)

        # Iterate until convergence or for max iterations
        for _ in range(self.max_iterations):
            # Assign samples to closest centroids (create clusters)
            clusters = self._create_clusters(centroids, data)

            prev_centroids = centroids
            # Calculate new centroids from the clusters
            centroids = self._calculate_centroids(clusters, data)

            # If no centroids have changed => convergence
            diff = centroids - prev_centroids
            if not diff.any():
                break

        self.kmeans_centroids = centroids
        return centroids

    # Predict the class of each sample
    def predict(self, data):

        # First check if we have determined the K-Means centroids
        if not self.kmeans_centroids.any():
            raise Exception("K-Means centroids have not yet been determined.\nRun the K-Means 'fit' function first.")

        clusters = self._create_clusters(self.kmeans_centroids, data)

        predicted_labels = self._get_cluster_labels(clusters, data)

        return predicted_labels


# Get the training data
# Import the Iris flower dataset
iris = datasets.load_iris()
train_data = np.array(iris.data)
train_labels = np.array(iris.target)
num_features = train_data.data.shape[1]

# Apply PCA to the data to reduce its dimensionality
pca = decomposition.PCA(n_components=3)
pca.fit(train_data)
train_data = pca.transform(train_data)

# *********************************************
# Apply K-Means Clustering MANUALLY
# *********************************************
# Create the K-Means Clustering Object 
unique_labels = np.unique(train_labels)
num_classes = len(unique_labels)
clf = KMeans(k=num_classes, max_iterations=3000)

centroids = clf.fit(train_data)

predicted_labels = clf.predict(train_data)


# Compute the training accuracy
Accuracy = 0
for index in range(len(train_labels)):
	# Cluster the data using K-Means
	current_label = train_labels[index]
	predicted_label = predicted_labels[index]

	if current_label == predicted_label:
		Accuracy += 1

Accuracy /= len(train_labels)

# Print stuff
print("Manual K-Means Classification Accuracy = ", Accuracy)

# *********************************************
# Apply K-Means Clustering using Sklearn
# *********************************************
# Create the K-Means Clustering Object 
unique_labels = np.unique(train_labels)
num_classes = len(unique_labels)
clf = cluster.KMeans(n_clusters=num_classes, max_iter=3000, n_init=10)

kmeans = clf.fit(train_data)


# Compute the training accuracy
Accuracy = 0
for index in range(len(train_labels)):
	# Cluster the data using K-Means
	current_sample = train_data[index].reshape(1,-1) 
	current_label = train_labels[index]
	predicted_label = kmeans.predict(current_sample)

	if current_label == predicted_label:
		Accuracy += 1

Accuracy /= len(train_labels)

# Print stuff
print("Sklearn K-Means Classification Accuracy = ", Accuracy)