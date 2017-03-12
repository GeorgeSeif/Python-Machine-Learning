import numpy as np
import random

# Split the data into train and test sets
def train_test_split(X, y, test_size=0.2):
	# First, shuffle the data
    train_data, train_labels = shuffle_data(X, y)

    # Split the training data from test data in the ratio specified in test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    x_train, x_test = train_data[:split_i], train_data[split_i:]
    y_train, y_test = train_labels[:split_i], train_labels[split_i:]

    return x_train, x_test, y_train, y_test

# Randomly shuffle the data
def shuffle_data(data, labels):
	if(len(data) != len(labels)):
		raise Exception("The given data and labels do NOT have the same length")

	combined = list(zip(data, labels))
	random.shuffle(combined)
	data[:], labels[:] = zip(*combined)
	return data, labels

# Calculate the distance between two vectors
def euclidean_distance(vec_1, vec_2):
	if(len(vec_1) != len(vec_2)):
		raise Exception("The two vectors do NOT have equal length")

	distance = 0
	for i in range(len(vec_1)):
		distance += pow((vec_1[i] - vec_2[i]), 2)

	return np.sqrt(distance)

def compute_mean_and_var(data):
	num_elements = len(data)
	total = [0] * data.shape[1]
	for sample in data:
		total = total + sample
	mean_features = np.divide(total, num_elements)

	total = [0] * data.shape[1]
	for sample in data:
		total = total + np.square(sample - mean_features)

	std_features = np.divide(total, num_elements)

	var_features = std_features ** 2

	return mean_features, var_features

def normalize_data(data):
	mean_features, var_features = compute_mean_and_var(data)
	std_features = np.sqrt(var_features)

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