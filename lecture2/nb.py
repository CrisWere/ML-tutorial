import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load the dataset, X = input data y = class labels
X,y = load_iris(return_X_y=True)

# Create the classifier object
classifier = GaussianNB()

scores = []
# Initialise the 5-fold cross-validation
kf = KFold(n_splits=5,shuffle=True)
for train_index,test_index in kf.split(X):
	# Generate the training and test partitions of X and Y for each iteration of CV	
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	classifier.fit(X_train,y_train)
	predictions = classifier.predict(X_test)
	score = metrics.f1_score(y_test,predictions,average="weighted")
	scores.append(score)

print("Average cross-validation score: {0}".format(np.average(scores)))
