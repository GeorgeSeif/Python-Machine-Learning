from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor
import numpy as np
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

clf = MLPRegressor(solver='sgd', learning_rate_init=0.0001, momentum=0.9, early_stopping=False, learning_rate='constant', hidden_layer_sizes=(15, 10, 5), max_iter=500, verbose=True)

# Load the Boston housing data set to regression training
# NOTE that this loads as a dictionairy
boston_dataset = load_boston()

train_data = np.array(boston_dataset.data)
train_labels = np.array(boston_dataset.target)
num_features = boston_dataset.data.shape[1]

# Randomly shuffle the data
combined = list(zip(train_data, train_labels))
random.shuffle(combined)
train_data[:], train_labels[:] = zip(*combined)

# Normalize the data to have zero-mean and unit variance
train_data = normalize_data(train_data)

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