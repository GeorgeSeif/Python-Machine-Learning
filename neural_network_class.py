from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import datasets
import random
import ml_helpers

clf = MLPClassifier(solver='sgd', learning_rate_init=0.01, momentum=0.9, early_stopping=False, learning_rate='constant', hidden_layer_sizes=(15, 10, 5), max_iter=500, verbose=True)

# Get the training data
# Import the Iris flower dataset
iris = datasets.load_iris()
train_data = np.array(iris.data)
train_labels = np.array(iris.target)
num_features = train_data.data.shape[1]

# Randomly shuffle the data
train_data, train_labels = ml_helpers.shuffle_data(train_data, train_labels)

# Normalize the training data
train_data = ml_helpers.normalize_data(train_data)

# Train the Neural Network on the data
clf.fit(train_data, train_labels)

# Compute the training accuracy
Accuracy = 0
for index in range(len(train_labels)):
	current_sample = train_data[index].reshape(1,-1) 
	current_label = train_labels[index]
	predicted_label = clf.predict(current_sample)

	if current_label == predicted_label:
		Accuracy += 1

Accuracy /= len(train_labels)

# Print stuff
print("Classification Accuracy = ", Accuracy)