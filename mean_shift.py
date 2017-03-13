import numpy as np
from sklearn import datasets, decomposition, cluster
import ml_helpers

class MeanShift:
    def __init__(self, radius=5, max_iters=100):
        self.radius = radius
        self.max_iters = max_iters
        self.mean_shift_centroids = []

    # Do Mean-Shift clustering and return the centroids of the clusters
    def fit(self, data):

        centroids = data
        
        curr_iter_count = 0
        while curr_iter_count in range(self.max_iters):
            new_centroids = []
            for i, _ in enumerate(centroids):
                within_window = []
                centroid = centroids[i]
                for sample in data:
                    if np.linalg.norm(sample - centroid) < self.radius:
                        within_window.append(sample)

                new_centroid = np.average(within_window, axis=0)
                new_centroids.append(new_centroid)

            prev_centroids = centroids

            optimized = True

            for i, _ in enumerate(centroids):
                if not np.array_equal(prev_centroids[i], new_centroids[i]):
                    optimized = False

                if not optimized:
                    break
                
            if optimized:
                unique_centroids = np.unique(new_centroids)
                break

            centroids = new_centroids
            curr_iter_count += 1

        self.mean_shift_centroids = unique_centroids
        
        return self.mean_shift_centroids

    # Predict the class of each sample
    def predict(self, data):
        # First check if we have determined the K-Means centroids
        if not self.mean_shift_centroids.any():
            raise Exception("Mean-Shift centroids have not yet been determined.\nRun the Mean-Shift 'fit' function first.")

        predicted_labels = np.zeros(len(data))
        for i in range(len(predicted_labels)):
        	curr_sample = data[i]
        	distances = [np.linalg.norm(curr_sample - centroid) for centroid in self.mean_shift_centroids]
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
# Apply Mean-Shift Clustering MANUALLY
# *********************************************
# Create the Mean-Shift Clustering Object 
clf = MeanShift(radius=5, max_iters=100)

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
print("Manual Mean-Shift Classification Accuracy = ", Accuracy)



# *********************************************
# Apply Mean-Shift Clustering using Scikit Learn
# *********************************************
# Create the Mean-Shift Clustering Object 
clf = cluster.MeanShift(bandwidth=5)

ms = clf.fit(train_data)

# Compute the training accuracy
Accuracy = 0
for index in range(len(train_labels)):
	# Cluster the data using K-Means
	current_sample = train_data[index].reshape(1,-1) 
	current_label = train_labels[index]
	predicted_label = ms.predict(current_sample)

	if current_label == predicted_label:
		Accuracy += 1

Accuracy /= len(train_labels)

# Print stuff
print("Sklean Mean-Shift Classification Accuracy = ", Accuracy)