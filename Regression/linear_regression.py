from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import numpy as np
import random
import matplotlib.pyplot as plt
import ml_helpers

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

weights = np.zeros(num_features + 1)

num_epochs = 3000
learning_rate = 0.001
reg = 0.1

final_predictions = [0] * len(train_labels)

# *********************************************
# Perform Linear Regression manually
# *********************************************
for curr_epoch in range(num_epochs):

	cost = 0
	gradient_error = 0
	for index, sample in enumerate(train_data):
		curr_label = train_labels[index]
		prediction = weights[1:].dot(sample) + weights[0]

		cost = cost + (prediction - curr_label) ** 2 + (reg * np.sum(weights ** 2))

		reg_array = np.append(0, np.full(num_features, reg))

		gradient_error = gradient_error + (curr_label - prediction)*np.append(1, sample) + (reg_array * weights)

		if curr_epoch == num_epochs - 1:
			final_predictions[index] = prediction

	weights = weights + learning_rate * (gradient_error / len(train_labels))

	cost = cost / 2

	print("Epoch # ", curr_epoch + 1, " with cost = ", cost)


# ***************************************************************
# Perform Linear Regression using Sklearn
# ***************************************************************
lm = LinearRegression()
lm.fit(train_data, train_labels)

# Plot outputs

# First plot the Sklearn Linear Regression
f = plt.figure(1)
plt.plot(train_labels, lm.predict(train_data),'ro')
plt.plot([0,50],[0,50], 'g-')
plt.xlabel('real')
plt.ylabel('predicted')
f.show()

# Now plot the manual Linear Regression
g = plt.figure(2)
plt.plot(train_labels, final_predictions,'ro')
plt.plot([0,50],[0,50], 'g-')
plt.xlabel('real')
plt.ylabel('predicted')
g.show()

# Keep the plots alive until we get a user input
print("Press any key to exit")
input()