import numpy as np
from sklearn import neighbors, datasets
from sklearn import decomposition
import random
from sklearn.neighbors import NearestNeighbors
import ml_helpers 

class KNN():
    def __init__(self, k=5):
        self.k = k

    # Do a majority vote among the neighbors
    def _majority_vote(self, neighbors, classes):
        max_count = 0
        most_common = None
        # Count class occurences among neighbors
        for c in np.unique(classes):
            # Count number of neighbors with class c
            count = len(neighbors[neighbors[:, 1] == c])
            if count > max_count:
                max_count = count
                most_common = c
        return most_common

    def predict(self, X_test, X_train, y_train):
        classes = np.unique(y_train)
        y_pred = []
        # Determine the class of each sample
        for test_sample in X_test:
            neighbors = []

            # Calculate the distance form each observed sample to the sample we wish to predict
            for j, observed_sample in enumerate(X_train):
                distance = ml_helpers.euclidean_distance(test_sample, observed_sample)
                label = y_train[j]

                # Add neighbor information
                neighbors.append([distance, label])
            neighbors = np.array(neighbors)

            # Sort the list of observed samples from lowest to highest distance and select the k first
            k_nearest_neighbors = neighbors[neighbors[:, 0].argsort()][:self.k]

            # Do a majority vote among the k neighbors and set prediction as the class receing the most votes
            label = self._majority_vote(k_nearest_neighbors, classes)
            y_pred.append(label)
        return np.array(y_pred)


# Get the training data
# Import the Iris flower dataset
iris = datasets.load_iris()
train_data = np.array(iris.data)
train_labels = np.array(iris.target)
num_features = train_data.data.shape[1]

# Normalize the training data
train_data = ml_helpers.normalize_data(train_data)

# Apply PCA to the data to reduce its dimensionality
pca = decomposition.PCA(n_components=2)
pca.fit(train_data)
train_data = pca.transform(train_data)


X_train, X_test, y_train, y_test = ml_helpers.train_test_split(train_data, train_labels, test_size=0.5)

# *********************************************
# Apply the KNN Classifier MANUALLY
# *********************************************
clf = KNN(k=5)
predicted_labels = clf.predict(X_test, X_train, y_train)

# Compute the training accuracy
Accuracy = 0
for index in range(len(y_test)):
	# Cluster the data using K-Means
	current_label = y_test[index]
	predicted_label = predicted_labels[index]

	if current_label == predicted_label:
		Accuracy += 1

Accuracy /= len(train_labels)

# Print stuff
print("Manual KNN Classification Accuracy = ", Accuracy)

# *********************************************
# Apply the KNN Classifier using Sklearn
# *********************************************
# Create the K-Means Clustering Object 
unique_labels = np.unique(train_labels)
num_classes = len(unique_labels)
clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='brute')

KNN = clf.fit(X_train, y_train)


# Compute the training accuracy
Accuracy = 0
for index in range(len(y_test)):
	# Cluster the data using K-Means
	current_sample = X_test[index].reshape(1,-1) 
	current_label = y_test[index]
	predicted_label = KNN.predict(current_sample)

	if current_label == predicted_label:
		Accuracy += 1

Accuracy /= len(train_labels)

# Print stuff
print("Sklean KNN Classification Accuracy = ", Accuracy)