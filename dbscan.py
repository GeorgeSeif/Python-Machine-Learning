import numpy as np
from sklearn import datasets, decomposition, cluster
import ml_helpers

class DBSCAN():
    def __init__(self, radius=1, min_samples=5):
        self.radius = radius
        self.min_samples = min_samples
        # List of arrays (clusters) containing sample indices
        self.clusters = []
        self.visited_samples = []
        # Hashmap {"sample_index": [neighbor1, neighbor2, ...]}
        self.neighbors = {}
        self.data = None   # Dataset

    # Return a list of neighboring samples
    # A sample_2 is considered a neighbor of sample_1 if the distance between
    # them is smaller than radiusi
    def _get_neighbors(self, sample_i):
        neighbors = []
        for _sample_i, _sample in enumerate(self.data):
            if _sample_i != sample_i and ml_helpers.euclidean_distance(self.data[sample_i], _sample) < self.radius:
                neighbors.append(_sample_i)
        return np.array(neighbors)

    # Recursive method which expands the cluster until we have reached the border
    # of the dense area (density determined by radius and min_samples)
    def _expand_cluster(self, sample_i, neighbors):
        cluster = [sample_i]
        # Iterate through neighbors
        for neighbor_i in neighbors:
            if not neighbor_i in self.visited_samples:
                self.visited_samples.append(neighbor_i)
                # Fetch the samples distant neighbors
                self.neighbors[neighbor_i] = self._get_neighbors(neighbor_i)
                # Make sure the neighbors neighbors are more than min_samples
                if len(self.neighbors[neighbor_i]) >= self.min_samples:
                    # Choose neighbors of neighbor except for sample
                    distant_neighbors = self.neighbors[neighbor_i][
                        np.where(self.neighbors[neighbor_i] != sample_i)]
                    # Add the neighbors neighbors as neighbors of sample
                    self.neighbors[sample_i] = np.concatenate(
                        (self.neighbors[sample_i], distant_neighbors))
                    # Expand the cluster from the neighbor
                    expanded_cluster = self._expand_cluster(
                        neighbor_i, self.neighbors[neighbor_i])
                    # Add expanded cluster to this cluster
                    cluster = cluster + expanded_cluster
            if not neighbor_i in np.array(self.clusters):
                cluster.append(neighbor_i)
        return cluster

    # Return the samples labels as the index of the cluster in which they are
    # contained
    def _get_cluster_labels(self):
        # Set default value to number of clusters
        # Will make sure all outliers have same cluster label
        labels = len(self.clusters) * np.ones(np.shape(self.data)[0])
        for cluster_i, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_i
        return labels

    # DBSCAN
    def predict(self, data):
        self.data = data
        n_samples = np.shape(self.data)[0]
        # Iterate through samples and expand clusters from them
        # if they have more neighbors than self.min_samples
        for sample_i in range(n_samples):
            if sample_i in self.visited_samples:
                continue
            self.visited_samples.append(sample_i)
            self.neighbors[sample_i] = self._get_neighbors(sample_i)
            if len(self.neighbors[sample_i]) >= self.min_samples:
                # Sample has more neighbors than self.min_samples => expand
                # cluster from sample
                new_cluster = self._expand_cluster(
                    sample_i, self.neighbors[sample_i])
                # Add cluster to list of clusters
                self.clusters.append(new_cluster)

        # Get the resulting cluster labels
        cluster_labels = self._get_cluster_labels()
        return cluster_labels

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
# Apply DBSCAN Clustering MANUALLY
# *********************************************
# Create the DBSCAN Clustering Object 
unique_labels = np.unique(train_labels)
num_classes = len(unique_labels)
clf = DBSCAN(radius=1, min_samples=5)

predicted_labels = clf.predict(train_data)


# Compute the training accuracy
Accuracy = 0
for index in range(len(train_labels)):
	# Cluster the data using DBSCAN
	current_label = train_labels[index]
	predicted_label = predicted_labels[index]

	if current_label == predicted_label:
		Accuracy += 1

Accuracy /= len(train_labels)

# Print stuff
print("Manual DBSCAN Classification Accuracy = ", Accuracy)

# *********************************************
# Apply DBSCAN Clustering using Sklearn
# *********************************************
# Create the DBSCAN Clustering Object 
unique_labels = np.unique(train_labels)
num_classes = len(unique_labels)
clf = cluster.DBSCAN(eps=1, min_samples=5)

predicted_labels = clf.fit_predict(train_data)


# Compute the training accuracy
Accuracy = 0
for index in range(len(train_labels)):
	# Cluster the data using DBSCAN
	current_label = train_labels[index]
	predicted_label = predicted_labels[index]

	if current_label == predicted_label:
		Accuracy += 1

Accuracy /= len(train_labels)

# Print stuff
print("Sklearn DBSCAN Classification Accuracy = ", Accuracy)