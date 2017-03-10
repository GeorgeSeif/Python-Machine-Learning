from sklearn import datasets
import numpy as np
import collections
from sklearn.naive_bayes import GaussianNB
import ml_helpers

def compute_gaussian_probability(val, mean, var):
	coeff = 1/(np.sqrt(2*np.pi) * np.sqrt(var))
	exponent = -1 * (((val - mean) ** 2) / (2 * var))
	return coeff * np.exp(exponent)

# Import the Iris flower dataset
iris = datasets.load_iris()
train_data = np.array(iris.data)
train_labels = np.array(iris.target)
num_features = train_data.data.shape[1]

# Get the class probabilities
unique_labels, class_probs = np.unique(train_labels, return_counts=True)

# Get the mean and variance of the features for ALL of the classes
mean_features, var_features = ml_helpers.compute_mean_and_var(train_data)

# Get the mean and variance of the features for EACH of the classes
class_mean_features = np.zeros((len(unique_labels), num_features))
class_var_features = np.zeros((len(unique_labels), num_features))
count = 0
for curr_label in unique_labels:
	class_data_indices = np.where(train_labels == curr_label)[0]

	temp_data = train_data[class_data_indices]
	class_mean_features[count, :], class_var_features[count, :] = ml_helpers.compute_mean_and_var(temp_data)
	count = count + 1


# ***************************************************************
# Apply the Naive Bayes Classifier MANUALLY
# ***************************************************************
predicted_classes = np.zeros(len(train_labels))
for sample_index, sample in enumerate(train_data):

	sample_probs = np.zeros(len(unique_labels))
	for curr_label_index, curr_label in enumerate(unique_labels):
		liklihood = np.prod(compute_gaussian_probability(sample, class_mean_features[curr_label_index], class_var_features[curr_label_index]))
		class_prior_prob = class_probs[curr_label_index]
		predictor_prior_prob = np.prod(compute_gaussian_probability(sample, mean_features[curr_label_index], var_features[curr_label_index]))
		sample_probs[curr_label_index] = (liklihood * class_prior_prob) / predictor_prior_prob

	predicted_classes[sample_index] = np.argmax(sample_probs)

# ***************************************************************
# Apply the Naive Bayes Classifier USING SCIKIT LEARN
# ***************************************************************
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Scikit Learn Classification Accuracy =", (iris.target == y_pred).sum()/ iris.data.shape[0])

# Compute the classification accuracy
Accuracy = 0
for label_index, curr_label in enumerate(train_labels):
	if curr_label == predicted_classes[label_index]:
		Accuracy = Accuracy + 1

Accuracy = Accuracy / len(train_labels)
print("Manual Classification Accuracy =", Accuracy)