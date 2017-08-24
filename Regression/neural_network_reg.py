from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor
import numpy as np
import random
import ml_helpers

clf = MLPRegressor(solver='sgd', learning_rate_init=0.0001, momentum=0.9, early_stopping=False, learning_rate='constant', hidden_layer_sizes=(15, 10, 5), max_iter=500, verbose=True)

# Load the Boston housing data set to regression training
# NOTE that this loads as a dictionairy
boston_dataset = load_boston()

train_data = np.array(boston_dataset.data)
train_labels = np.array(boston_dataset.target)
num_features = boston_dataset.data.shape[1]

# Randomly shuffle the data
train_data, train_labels = ml_helpers.shuffle_data(train_data, train_labels)

# Normalize the data to have zero-mean and unit variance
train_data = ml_helpers.normalize_data(train_data)

# Train the Neural Network on the data
clf.fit(train_data, train_labels)

# Compute the average error
Average_error = 0
for index, sample in enumerate(train_data):
	curr_sample = sample.reshape(1,-1) 
	curr_label = train_labels[index]
	predicted_label = clf.predict(curr_sample)
	Average_error += np.abs(predicted_label - curr_label)

Average_error /= len(train_labels) 

# Print stuff
print("Average Error = ", Average_error)