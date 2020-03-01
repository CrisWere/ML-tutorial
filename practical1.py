import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing
from matplotlib import pyplot


# Load the dataset, X = input data y = class labels
data = pd.read_csv("checkerboard.csv")
print(data)

X = data.iloc[:,:-1].values
#X = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values
# Class labels are strings in this case, and have to be converted to integers
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Create the classifier object
classifier = RandomForestClassifier()
#classifier = KNeighborsClassifier()

# Dictionary that contains the values for the parameter sweep
# Params dictionary of random forest
param_grid = dict(max_depth=[2,3,4],n_estimators = [100, 200, 500])
# Params dictionary of KNN
#param_grid = dict(n_neighbors=[2,3,4,5,10])

scores = []
preds = []
actual_labels = []
# Initialise the 5-fold cross-validation
kf = KFold(n_splits=5,shuffle=True)
for train_index,test_index in kf.split(X):
	# Generate the training and test partitions of X and Y for each iteration of CV	
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	# Increasing the value of the verbose parameter will give more messags of the internal grid search process
	# Increasing n_jobs will tell it to use multiple cores to parallelise the computation	
	grid_search = GridSearchCV(classifier,param_grid=param_grid,cv=5,scoring="f1",verbose=0,n_jobs=4)
	grid_search.fit(X_train,y_train)

	# Printing the values of the parameters chosen by grid search
	estimator = grid_search.best_estimator_
	print("Chosen max depth: {0}".format(estimator.max_depth))
	print("Chosen number of trees: {0}".format(estimator.n_estimators))
	#print("Number of neighbours: {0}".format(estimator.n_neighbors))

	# Predicting the test data with the optimised models
	predictions = estimator.predict(X_test)
	score = metrics.f1_score(y_test,predictions)
	scores.append(score)

	# Extract the probabiliites of predicting the 2nd class, which will use to generate the PR curve
	probs =estimator.predict_proba(X_test)[:,1]
	preds.extend(probs)
	actual_labels.extend(y_test)
	

# Report the overall F1 score
print("F1 score: {0}".format(np.average(scores)))

prec, recall, _ = metrics.precision_recall_curve(actual_labels, preds)
print("AUPRC score: {0}".format(metrics.auc(recall,prec)))
# Generate the PR curve
pyplot.plot(recall, prec, marker='.')
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.savefig('prcurve.pdf')
pyplot.close()

