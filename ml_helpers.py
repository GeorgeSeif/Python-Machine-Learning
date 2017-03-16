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

# Compute the mean and variance of each feature of a data set
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

# Normalize data by subtracting mean and dividing by standard deviation
def normalize_data(data):
	mean_features, var_features = compute_mean_and_var(data)
	std_features = np.sqrt(var_features)

	for index, sample in enumerate(data):
		data[index] = np.divide((sample - mean_features), std_features) 

	return data

# Divide dataset based on if sample value on feature index is larger than
# the given threshold
def divide_on_feature(X, feature_i, threshold):
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])

# Return random subsets (with replacements) of the data
def get_random_subsets(X, y, n_subsets, replacements=True):
    n_samples = np.shape(X)[0]
    # Concatenate x and y and do a random shuffle
    X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets = []

    # Uses 50% of training samples without replacements
    subsample_size = n_samples // 2
    if replacements:
        subsample_size = n_samples      # 100% with replacements

    for _ in range(n_subsets):
        idx = np.random.choice(range(n_samples), size=np.shape(range(subsample_size)), replace=replacements)
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, -1]
        subsets.append([X, y])
    return subsets

# Calculate the entropy of label array y
def calculate_entropy(y):
    log2 = lambda x: np.log(x) / np.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy

# Returns the mean squared error between y_true and y_pred
def mean_squared_error(y_true, y_pred):
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse

# The sigmoid function
def sigmoid(val):
	return np.divide(1, (1 + np.exp(-1*val)))

# The derivative of the sigmoid function
def sigmoid_gradient(val):
    return sigmoid(val) * (1 - sigmoid(val))

# Compute the covariance matrix of an array
def compute_cov_mat(data):
	# Compute the mean of the data
	mean_vec = np.mean(data, axis=0)

	# Compute the covariance matrix
	cov_mat = (data - mean_vec).T.dot((data - mean_vec)) / (data.shape[0]-1)

	return cov_mat


# Perform PCA dimensionality reduction
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

# 1D Gaussian Function
def gaussian_1d(val, mean, standard_dev):
	coeff = 1 / (standard_dev * np.sqrt(2 * np.pi))
	exponent = (-1 * (val - mean) ** 2) / (2 * (standard_dev ** 2))
	gauss = coeff * np.exp(exponent)
	return gauss

# 2D Gaussian Function
def gaussian_2d(x_val, y_val, x_mean, y_mean, x_standard_dev, y_standard_dev):
	x_gauss = gaussian_1d(x_val, x_mean, x_standard_dev)
	y_gauss = gaussian_1d(y_val, y_mean, y_standard_dev)
	gauss = x_gauss * y_gauss
	return gauss
