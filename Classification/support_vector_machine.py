import numpy as np
from sklearn import datasets
from sklearn import svm
import random
import ml_helpers

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