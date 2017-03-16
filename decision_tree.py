import numpy as np
from sklearn import datasets, decomposition
from sklearn.datasets import load_boston
import ml_helpers
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

class DecisionNode():
    # Class that represents a decision node or leaf in the decision tree
    # Parameters:
    # -----------
    # feature_i: int
    #     Feature index which we want to use as the threshold measure.
    # threshold: float
    #     The value that we will compare feature values at feature_i against to 
    #     determine the prediction.
    # value: float
    #     The class prediction if classification tree, or float value if regression tree.
    # true_branch: DecisionNode
    #     Next decision node for samples where features value met the threshold.
    # false_branch: DecisionNode
    #     Next decision node for samples where features value did not meet the threshold.

    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Value if the node is a leaf in the tree
        self.true_branch = true_branch      # 'Left' subtree
        self.false_branch = false_branch    # 'Right' subtree


# Super class of RegressionTree and ClassificationTree
class DecisionTree(object):
    # Super class of RegressionTree and ClassificationTree.
    # Parameters:
    # -----------
    
    # min_samples_split: int
    #     The minimum number of samples needed to make a split when building a tree.
    # min_impurity: float
    #     The minimum impurity required to split the tree further. 
    # max_depth: int
    #     The maximum depth of a tree.

    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf"), loss=None):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        # If y is nominal
        self.one_dim = None

    def fit(self, X, y, loss=None):
        # Build tree
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)

        self.loss=None

    def _build_tree(self, X, y, current_depth=0):

        largest_impurity = 0
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data

        expand_needed = len(np.shape(y)) == 1
        if expand_needed:
            y = np.expand_dims(y, axis=1)

        # Add y as last column of X
        X_y = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    Xy1, Xy2 = ml_helpers.divide_on_feature(X_y, feature_i, threshold)
                    
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2)

                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {
                                "feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],
                                "lefty": Xy1[:, n_features:],
                                "rightX": Xy2[:, :n_features],
                                "righty": Xy2[:, n_features:]
                                }

        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)

    # Do a recursive search down the tree and make a predict of the data sample by the
    # value of the leaf that we end up at
    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        # If we have a value => return prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # Test subtree
        return self.predict_value(x, branch)

    # Classify samples one by one and return the set of labels
    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print (tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print ("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class RegressionTree(DecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        _, var_tot = ml_helpers.compute_mean_and_var(y)
        _, var_1 = ml_helpers.compute_mean_and_var(y1)
        _, var_2 = ml_helpers.compute_mean_and_var(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        # Calculate the variance reduction
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

        return sum(variance_reduction)

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)

class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        # Calculate information gain
        p = len(y1) / len(y)
        entropy = ml_helpers.calculate_entropy(y)
        info_gain = entropy - p * \
            ml_helpers.calculate_entropy(y1) - (1 - p) * \
            ml_helpers.calculate_entropy(y2)

        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)


def main():

	# **************************************************************
	# Apply the Decision Tree for Classification Manually
	# **************************************************************
	# Get the training data
	# Import the Iris flower dataset
	iris = datasets.load_iris()
	train_data = np.array(iris.data)
	train_labels = np.array(iris.target)
	num_features = train_data.data.shape[1]

	# Randomly shuffle the data
	train_data, train_labels = ml_helpers.shuffle_data(train_data, train_labels)

	# Apply PCA to the data to reduce its dimensionality
	pca = decomposition.PCA(n_components=4)
	pca.fit(train_data)
	train_data = pca.transform(train_data)


	X_train, X_test, y_train, y_test = ml_helpers.train_test_split(train_data, train_labels, test_size=0.4)

	clf = ClassificationTree()

	clf.fit(X_train, y_train)

	predicted_labels = clf.predict(X_test)

	# Compute the testing accuracy
	Accuracy = 0
	for index in range(len(predicted_labels)):
		current_label = y_test[index]
		predicted_label = predicted_labels[index]

		if current_label == predicted_label:
			Accuracy += 1

	Accuracy /= len(train_labels)

	# Print stuff
	print("Manual Decision Tree Classification Accuracy = ", Accuracy)


	# **************************************************************
	# Apply the Decision Tree for Classification using Sklearn
	# **************************************************************

	clf = DecisionTreeClassifier(criterion="gini", splitter="best")

	clf.fit(X=X_train, y=y_train)

	predicted_labels = clf.predict(X_test)

	# Compute the testing accuracy
	Accuracy = 0
	for index in range(len(predicted_labels)):
		current_label = y_test[index]
		predicted_label = predicted_labels[index]

		if current_label == predicted_label:
			Accuracy += 1

	Accuracy /= len(train_labels)

	# Print stuff
	print("Sklearn Decision Tree Classification Accuracy = ", Accuracy)


	# **************************************************************
	# Apply the Decision Tree for Regression Manually
	# **************************************************************
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

	X_train, X_test, y_train, y_test = ml_helpers.train_test_split(train_data, train_labels, test_size=0.4)

	clf = RegressionTree()

	clf.fit(X_train, y_train)

	predicted_values = clf.predict(X_test)

	mse = ml_helpers.mean_squared_error(y_test, predicted_values)

	print ("Manual Decision Tree Regression Mean Squared Error:", mse)

	# Now plot the manual Linear Regression
	g = plt.figure(1)
	plt.plot(y_test, predicted_values,'ro')
	plt.plot([0,50],[0,50], 'g-')
	plt.xlabel('real')
	plt.ylabel('predicted')
	g.show()

	# **************************************************************
	# Apply the Decision Tree for Regression using Sklearn
	# **************************************************************
	clf = DecisionTreeRegressor(criterion="mse", splitter="best")

	clf.fit(X_train, y_train)

	predicted_values = clf.predict(X_test)

	mse = ml_helpers.mean_squared_error(y_test, predicted_values)

	print ("Sklearn Decision Tree Regression Mean Squared Error:", mse)

	# Now plot the manual Linear Regression
	g = plt.figure(2)
	plt.plot(y_test, predicted_values,'ro')
	plt.plot([0,50],[0,50], 'g-')
	plt.xlabel('real')
	plt.ylabel('predicted')
	g.show()

	# Keep the plots alive until we get a user input
	print("Press any key to exit")
	input()

if __name__ == "__main__":
    main()