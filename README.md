# Python Machine Learning

## Description

A set of machine learing algorithms implemented in Python 3.5. Please also see my related repository for [Python Data Science](https://github.com/GeorgeSeif/Data-Science-Python) which contains various data science scripts for data analysis and visualisation.

Programs can one of three implementations:

1. Algorithm is implemented from scratch in Python *
2. Algorithm is implemented using Scikit Learn * *
3. Algorithm is implemented both ways * * *

The included programs are:

- **Regression:**
	- Linear Regression * * *
	- Neural Network Regression * *
	- Decision Tree Regression * * * 

- **Classification:**
	- Logistic Regression for Classification * * *
	- Logistic Regression for Classification with PCA * * *
	- Naive Bayes Classification * * *
	- Support Vector Machine Classification * *
	- Neural Network Classification * *
	- Decision Tree Classification * * *
	- Random Forest Classification * * *

- **Clustering:**
	- K-Means Clustering * * *
	- K-Nearest-Neighbor * * * 
	- Mean-Shift Clustering * * *
	- K-Mediods Clustering *
	- DBSCAN Clustering * * * 

## Information

Here are the descriptions of the above machine learning algorithms

### Linear and Logistic Regression

Regression is a technique used to model and analyze the relationships between variables and often times how they contribute and are related to producing a particular outcome together. Beginning with the simple case, _Single Variable Linear Regression_ is a technique used to model the relationship between a single input independant variable (feature variable) and an output dependant variable using a linear model i.e a line. The more general case is _Multi Variable Linear Regression_ where a model is created for the relationship between multiple independant input variables (feature variables) and an output dependant variable. The model remains linear in that the output is a linear combination of the input variables. There is a third most general case called _Polynomial Regression_ where the model now becomes a _non-linear_ combination of the feature variables; this however requires knowledge of how the data relates to the output. For all of these cases of the regression the output variable is a real-number (rather than a class category). We can also do logistic regression where instead of predicting a real-number, we predict the class or group that the input variable represent. This can be done by modifying the regression training such that the error is computed as the probability that the current example belongs to a particular class. This can be done simply by taking the sigmoid of the regular linear regression result and using a one vs. all scheme, or simple applying the Softmax function. Regression models can be trained using Stochastic Gradient Descent (SGD). Regression is fast to model and is particularly useful when the relationship to be modelled is not extremely complex (if it is complex, better off using something like a neural network).

### Neural Networks

A _Neural Network_ consists of an interconnected group of nodes called _neurons_. The input feature variables from the data are passed to these neurons as a multi-variable linear combination, where the values multiplied by each feature variable are known as _weights_. A non-linearity is then applied to this linear combination which gives the neural network the ability to model complex non-linear relationships. A neural network can have multiple layers where the output of one layer is passed to the next one in the same way. At the output, there is generally no non-linearity applied. Neural Networks are trained using Stochastic Gradient Descent (SGD) and the backpropagation algorithm. Neural networks can be used for either classification or regression in the same way as the linear and logistic regressions' description above. Since neural networks can have many layers with non-linearities, they are most suitable for modelling complex non-linear relationships in which taking the time to train them properly is worthwhile. 

### Naive Bayes

Naive Bayes is a classification technique based on Bayes' Theorem with the assumption that all feature variables are independant. Thus a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. Naive Bayes classifier combines 3 terms to compute the probability of a class: the class probability in the dataset, multiplied by the probability of the example feature variables occuring given the current class, divided by the probability of those particular example feature variables occuring in general. To compute the probability of particular feature variables occuring there are 3 main optional techniques. One can assume that the value of a particular variable is _Gaussian_ distributed which can be a common case and thus this method is useful when the variables are real numbers. _Multinomial_ division is good for feature variables that are categorical as it computes the probability based on histogram bins. The final option is to use a _Bernouli_ probability model when the data is binary. Naive Bayes is simplistic and easy to use yet can outperform other more complex classification algorithms. Is has fast computation and thus is well suited for application on large datasets.

### Support Vector Machines

Support Vector Machines (SVMs) are a learning algorithm mainly used for classification where the learned model is a set of hyperplanes that seperate the data into the respective classes. The hyperplanes are selected by learning the parameters that maximize the distance from the support vectors to the hyperplane, where the support vectors are the closest vectors to the plane. If hyper-plane having low margin is selected, then there is high chance of miss-classification. If the data is not linearly seperable in the current dimensional space, a _kernel function_ can be used to transform the data into a higher dimensional space where it is seperable via a linear hyperplance. SVMs have the strong advantage that they are highly effective in high dimensional feature space and at modelling complex relationships due to the fact that their kernal functions can be specified. Specifying a particular kernel function may require knowledge of the data, however there are some common default ones that often work well in practice. Additionally, the training of the SVMs only uses a very small subset of the data i.e the support vectors and is thus very memory efficient. The disadvantage to SVMs is there long training time as effective training involves complex quadratic programming algorithms. 

### Decision Trees and Random Forests

Beginning with the base case, a _Decision Tree_ is an intuitive model where by one traverses down the branches of the tree and selects the next branch to go down based on a decision at a node. The model- (or tree-) building aspect of decision tree classification algorithms are composed of 2 main tasks: tree induction and tree pruning. Tree induction is the task of taking a set of pre-classified instances as input, deciding which attributes are best to split on, splitting the dataset, and recursing on the resulting split datasets until all training instances are categorized. While building the tree, the goal is to split on the attributes which create the purest child nodes possible, which would keep to a minimum the number of splits that would need to be made in order to classify all instances in our dataset. This purity is generally measured by one of a number of different attribute selection measures. Purity is measured by the concept of information gain, which relates to how much would need to be known about a previously-unseen instance in order for it to be properly classified. In practice, this is measured by comparing entropy, or the amount of information needed to classify a single instance of a current dataset partition, to the amount of information to classify a single instance if the current dataset partition were to be further partitioned on a given attribute. 

Because of the nature of training decision trees they can be prone to major overfitting. A completed decision tree model can be overly-complex, contain unnecessary structure, and be difficult to interpret. Tree pruning is the process of removing the unnecessary structure from a decision tree in order to make it more efficient, more easily-readable for humans, and more accurate as well. Tree pruning is generally done by compressing part of the tree into less strict and rigid decision boundaries into ones that are more smooth and generalize better. Scikit learn comes with a biult in tool to visualize the full decision tree, including the specific decision boundaries set by each node. 

Random Forests are simply an ensemble of decision trees. The input vector is run through multiple decision trees. For regression, the output value of all the trees is averaged; for classification a voting scheme is used to determine the final class.



### Clustering

Clustereing is an unsupervised learning technique which involves grouping the data points based on their intrinsic feature variable similarity. A number of clustering algorithms are outlined below:

- **K-Means Clustering:** First, randomly select a number of classes/groups and their respective center points (the center points are vectors of the same length as each data point vector). Group the data points by computing the distance between the point and each group center, then picking the closest one. Based on the grouped points, recompute the group center by taking the mean of all the vectors in the group. Repeat these steps for a set number of iterations and optionally a set number of randomizations of starting points.
- **K-Mediods Clustering:** This is the same as K-Means except instead of recomputing the center points using the mean we use the median vector of the group. This method is less sensitive to outliers but is much slower for large datasets as sorting is required. 
- **K-Nearest-Neighbor:** In KNN, "K" represents the number of _voting points_. To group a data vector, the distance from that vector to all of the vectors in a labelled dataset are computed (commonly euclidean distance). Then the vectors from the labelled dataset with the closest distance to the data vector are chosen for voting. The data vector is then classified into the group for which the majority of the K-Nearest vectors (the voting group) belong to. The advantage of using KNN is that it is generally fast and easy to interpret. The drawback is that it is generally not as accurate as other clustering methods, especially with outliers.  
- **Mean-Shift Clustering:** Consider a set of points in two-dimensional space. Assume a circular window centered at C and having radius r as the kernel. Mean shift is a hill climbing algorithm which involves shifting this kernel iteratively to a higher density region until convergence. Every shift is defined by a mean shift vector. The mean shift vector always points toward the direction of the maximum increase in the density. First, a starting point is randomly selected. At every iteration the kernel is shifted to the centroid or the mean of the points within it. The method of calculating this mean depends on the choice of the kernel. In this case if a Gaussian kernel is chosen instead of a flat kernel, then every point will first be assigned a weight which will decay exponentially as the distance from the kernel's center increases. At convergence, there will be no direction at which a shift can accommodate more points inside the kernel. These candidate points are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids. In contrast to K-means clustering there is no need to select the number of clusters as mean-shift automatically discovers this. Then fact that the cluster centers converge towards the points of maximum density is also quite desirable as it is intuitivaley understandable. The drawback is that the selection of the window size/radius "r" is non-trivial.
- **DBSCAN Clustering:** DBSCAN is a density based clustered algorithm similar to mean-shift. It starts with an arbitrary starting point that has not been visited. The neighborhood of this point is extracted using ε (All points which are within the ε distance are neighborhood). If there are sufficient number of points within this neighbourhood then the clustering process starts and that point is marked as visited else this point is labeled as noise (Later this point can become the part of the cluster). If a point is found to be a part of the cluster then its ε neighborhood is also the part of the cluster and the above procedure is repeated for all ε neighborhood points. This is repeated until all points in the cluster is determined. A new unvisited point is retrieved and processed, leading to the discovery of a further cluster or noise. This process continues until all points are marked as visited. The advantages of DBSCAN are that it does not require a pe-set number of clusters, it identifies outliers as noises, and it is able to find arbitrarily size and arbitrarily shaped clusters. However, it does have the drawback that it has been found to perform worse than other clustering algorithms when using very high dimensional data; it also can fail in cases of varying density clusters.

### Scikit Learn Machine Learning Cheat Sheet

![alt text](https://github.com/GeorgeSeif/Python-Machine-Learning/blob/master/ml_cheatsheet.png)



## Helpers
In addition the the main algorithm files, we have the following set of helper functions in the "ml_helpers.py" file:

1. Train and Test data splitting
2. Random shuffling of data
3. Compute Euclidean Distance
4. Compute Mean and Variance of features
5. Normalize data
6. Divide dataset based on feature threshold
7. Retrieve a random subset of the data with a random subset of the features
8. Compute entropy
9. Compute Mean Squared Error
10. Sigmoid function
11. Derivative of the sigmoid function
12. Compute the covariance matrix
13. Perform PCA dimensionality reduction
14. Gaussian function 1D
15. Gaussian function 2D

## Requirements
1. Python 3.5
2. Numpy
3. Scipy
4. Scikit Learn
5. Matplotlib

## Installation
The above packages can be installed by running the commands listed in the "install.txt" file
