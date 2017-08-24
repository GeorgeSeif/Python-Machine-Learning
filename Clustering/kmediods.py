import numpy as np
from sklearn import datasets, decomposition, cluster
import ml_helpers
import sys

class KMediods():
    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations
        self.kmediods_centroids = []

    # Initialize the centroids from the given data points
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
        	curr_cluster = data[cluster]
        	smallest_dist = float("inf")
        	for point in curr_cluster:
        		total_dist = np.sum(ml_helpers.euclidean_distance(curr_cluster, [point] * len(curr_cluster)))
        		if total_dist < smallest_dist:
        			centroids[i] = point
        return centroids

    # Classify samples as the index of their clusters
    def _get_cluster_labels(self, clusters, data):
        # One prediction for each sample
        y_pred = np.zeros(np.shape(data)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # Do K-Mediods clustering and return the centroids of the clusters
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

        self.kmediods_centroids = centroids
        return clusters


    # Predict the class of each sample
    def predict(self, data):
        # First check if we have determined the K-Mediods centroids
        if not self.kmediods_centroids.any():
            raise Exception("Mean-Shift centroids have not yet been determined.\nRun the Mean-Shift 'fit' function first.")

        predicted_labels = np.zeros(len(data))
        for i in range(len(predicted_labels)):
        	curr_sample = data[i]
        	distances = [np.linalg.norm(curr_sample - centroid) for centroid in self.kmediods_centroids]
        	label = (distances.index(min(distances)))
        	predicted_labels[i] = label
        	
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
# Apply K-Mediods Clustering MANUALLY
# *********************************************
# Create the K-Mediods Clustering Object 
unique_labels = np.unique(train_labels)
num_classes = len(unique_labels)
clf = KMediods(k=num_classes, max_iterations=3000)

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
print("Manual K-Mediods Classification Accuracy = ", Accuracy)