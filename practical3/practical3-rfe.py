import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
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
selector = RFE(classifier, n_features_to_select = 10, step=1)

# A pipeline chains two algorithms together so that the training process for both can be done in a single step and data is passed automatically from one to the other
pipeline = Pipeline([("RFE", selector), ("classifier",classifier)])

# Dictionary that contains the values for the parameter sweep, uncomment the second version for a wider parameter sweep
param_grid = dict(RFE__n_features_to_select=[2,3,4,5], classifier__max_depth=[10], classifier__n_estimators=[100])
#param_grid = dict(RFE__n_features_to_select=[2,3,4,5], classifier__max_depth=[2, 3, 4, 10], classifier__n_estimators=[100, 200, 500])


scores = []
# Initialise the 5-fold cross-validation
kf = KFold(n_splits=5,shuffle=True)
for train_index,test_index in kf.split(X):
	# Generate the training and test partitions of X and Y for each iteration of CV	
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	# Increasing the value of the verbose parameter will give more messags of the internal grid search process
	# Increasing n_jobs will tell it to use multiple cores to parallelise the computation	
	grid_search = GridSearchCV(pipeline,param_grid=param_grid,cv=5,scoring="f1",verbose=0,n_jobs=4)
	grid_search.fit(X_train,y_train)

	# Printing the values of the parameters chosen by grid search
	estimator = grid_search.best_estimator_
	print("Number of selected features {0}".format(estimator.named_steps['RFE'].n_features_to_select))
	print("Selected features {0}".format(np.where(estimator.named_steps['RFE'].support_)))
	print("Max depth {0}".format(estimator.named_steps['classifier'].max_depth))
	print("Number of trees {0}".format(estimator.named_steps['classifier'].n_estimators))

	# Predicting the test data with the optimised models
	predictions = estimator.predict(X_test)
	score = metrics.f1_score(y_test,predictions)
	print("F1 score for this test set: {0}".format(score))
	scores.append(score)

# Report the overall F1 score
print("Overall F1 score: {0}".format(np.average(scores)))


