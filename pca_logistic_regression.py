import csv
import numpy as np
from sklearn import datasets
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
import ml_helpers


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

def sigmoid(val):
	return np.divide(1, (1 + np.exp(-1*val)))

def compute_cov_mat(data):
	# Compute the mean of the data
	mean_vec = np.mean(data, axis=0)

	# Compute the covariance matrix
	cov_mat = (data - mean_vec).T.dot((data - mean_vec)) / (data.shape[0]-1)

	return cov_mat


def pca(data, exp_var_percentage=95):

	# Compute the covariance matrix
	cov_mat = compute_cov_mat(data)

	# Compute the eigen values and vectors of the covariance matrix
	eig_vals, eig_vecs = np.linalg.eig(cov_mat)

	# Make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs.sort(key=lambda x: x[0], reverse=True)

	# Only keep a certain number of eigen vectors based on the "explained variance percentage"
	# which tells us how much information (variance) can be attributed to each of the principal components
	tot = sum(eig_vals)
	var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
	cum_var_exp = np.cumsum(var_exp)

	num_vec_to_keep = 0

	for index, percentage in enumerate(cum_var_exp):
		if percentage > exp_var_percentage:
			num_vec_to_keep = index + 1
			break

	# Compute the projection matrix based on the top eigen vectors
	proj_mat = eig_pairs[0][1].reshape(4,1)
	for eig_vec_idx in range(1, num_vec_to_keep):
		proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(4,1)))

	# Project the data 
	pca_data = data.dot(proj_mat)

	return pca_data

# Get the training data
# Import the Iris flower dataset
iris = datasets.load_iris()
train_data = np.array(iris.data)
train_labels = np.array(iris.target)

# Randomly shuffle the data
train_data, train_labels = ml_helpers.shuffle_data(train_data, train_labels)

# Normalize the training data
train_data = ml_helpers.normalize_data(train_data)

# ***************************************************************
# Apply PCA MANUALLY
# ***************************************************************
train_data = ml_helpers.pca(train_data, 90)

# ***************************************************************
# Apply PCA using Sklean
# ***************************************************************
# pca = decomposition.PCA(n_components=2)
# pca.fit(train_data)
# train_data = pca.transform(train_data)

num_epochs = 1000
learning_rate = 0.01

unique_labels = np.unique(train_labels)
num_classes = len(unique_labels)

num_features = train_data.data.shape[1]
weights = np.zeros((num_features + 1, num_classes))

final_predictions = [0] * len(train_labels)

# Perform Logistic Regression manually
# We will use a ONE vs ALL scheme
for curr_epoch in range(num_epochs):

	cost = 0
	gradient_error = np.zeros((num_features + 1, num_classes))
	for index, sample in enumerate(train_data):

		curr_label = int(train_labels[index])
		one_hot_index = np.where(unique_labels == curr_label)
		curr_one_hot_labels = np.zeros(num_classes)
		curr_one_hot_labels[one_hot_index] = 1

		class_predictions = np.zeros(num_classes)

		for class_index in range(num_classes):
			class_predictions[class_index] = weights[1:, class_index].dot(sample) + weights[0, class_index]
			
		
		class_predictions = sigmoid(class_predictions)

		cost = cost +  -1*np.sum((curr_one_hot_labels.dot(np.log(class_predictions)) + (1 - curr_one_hot_labels).dot(np.log(1 - class_predictions))))  + (reg * np.sum(weights ** 2, axis=1))

		reg_array = np.append(0, np.full(num_features, reg))


		for class_index in range(num_classes):
			gradient_error[:, class_index] = gradient_error[:, class_index] + (curr_one_hot_labels[class_index] - class_predictions[class_index])*np.append(1, sample) + (reg_array * weights[:, class_index])

		if curr_epoch == num_epochs - 1:
			final_predictions[index] = class_predictions

	weights = weights + learning_rate * (gradient_error / len(train_labels))

	print("Epoch # ", curr_epoch + 1, " with cost = ", cost)


# Perform Logistic Regression using Sklean
lm = LogisticRegression()
lm.fit(train_data, train_labels)
sklearn_predictions = lm.predict(train_data)

# Test out the training accuracy
Accuracy = 0
for pred_idx, pred in enumerate(final_predictions):
	pred_class_index = np.argmax(pred)
	# pred_class_index = list(unique_labels).index(pred) # FOR THE Sklean LOGISTIC REGRESSION. ALSO SUB IN "predictions" instead of "sklearn_predictions"

	curr_label = int(train_labels[pred_idx])
	one_hot_index = list(unique_labels).index(curr_label)
	
	if pred_class_index == one_hot_index:
		Accuracy = Accuracy + 1

Accuracy = Accuracy / len(final_predictions)

print("Final classification accuracy = ", Accuracy)