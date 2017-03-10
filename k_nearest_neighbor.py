import numpy as np
from sklearn import datasets
from sklearn import decomposition
import random

def normalize_data(data):
	num_elements = len(data)
	total = [0] * data.shape[1]
	for sample in data:
		total = total + sample
	mean_features = np.divide(total, num_elements)

	total = [0] * data.shape[1]
	for sample in data:
		total = total + np.square(sample - mean_features)

	std_features = np.divide(total, num_elements)

	for index, sample in enumerate(data):
		data[index] = np.divide((sample - mean_features), std_features) 

	return data

# Calculate the distance between two vectors
def euclidean_distance(vec_1, vec_2):
    distance = 0
    for i in range(len(vec_1)):
        distance += pow((vec_1[i] - vec_2[i]), 2)

    return np.sqrt(distance)

# Split the data into train and test sets
def train_test_split(X, y, test_size=0.2):
    # Randomly shuffle the data
    combined = list(zip(train_data, train_labels))
    random.shuffle(combined)
    train_data[:], train_labels[:] = zip(*combined)

    # Split the training data from test data in the ratio specified in test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    x_train, x_test = train_data[:split_i], train_data[split_i:]
    y_train, y_test = train_labels[:split_i], train_labels[split_i:]

    return x_train, x_test, y_train, y_test

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
                distance = euclidean_distance(test_sample, observed_sample)
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
train_data = normalize_data(train_data)

# Apply PCA to the data to reduce its dimensionality
pca = decomposition.PCA(n_components=2)
pca.fit(train_data)
train_data = pca.transform(train_data)


X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.5)

clf = KNN(k=3)
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
print("Classification Accuracy = ", Accuracy)