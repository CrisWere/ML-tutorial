import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load the dataset, X = input data y = class labels
X,y = load_iris(return_X_y=True)

# Create the classifier object
classifier = DecisionTreeClassifier()

# Dictionary that contains the values for the parameter sweep
param_grid = dict(max_depth=[2,3,4,5,10])

scores = []
# Initialise the 5-fold cross-validation
kf = KFold(n_splits=5,shuffle=True)
for train_index,test_index in kf.split(X):
	# Generate the training and test partitions of X and Y for each iteration of CV	
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	grid_search = GridSearchCV(classifier,param_grid=param_grid,cv=5,scoring="f1_weighted")
	grid_search.fit(X_train,y_train)

	estimator = grid_search.best_estimator_
	print("Chosen max depth: {0}".format(estimator.max_depth))

	predictions = estimator.predict(X_test)
	score = metrics.f1_score(y_test,predictions,average="weighted")
	scores.append(score)
	print("Score of best model in the test set: {0}".format(score))

print("Average cross-validation score: {0}".format(np.average(scores)))
