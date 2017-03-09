import numpy as np
from sklearn import datasets
from sklearn import svm
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

# Get the training data
# Import the Iris flower dataset
iris = datasets.load_iris()
train_data = np.array(iris.data)
train_labels = np.array(iris.target)
num_features = train_data.data.shape[1]

# Create the SVM classification object
# "kernel" --> Identifies the kernel type to be used in the algorithm.  
# "C"	   --> This is the slackness which controls how much weight the error will have in training the SVM; 
# Larger "C" gives more penalty too the errors, forcing the SVM to learn them and taking longer to train
clf = svm.SVC(kernel='linear', C = 1.0)

# Train the SVM on the data
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