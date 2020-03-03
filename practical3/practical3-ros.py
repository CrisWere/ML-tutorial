import arff
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
import time

start = time.perf_counter()

print("Loading training set")
data = np.array(arff.load(open('TrainFold00w4.arff', 'r'))["data"])
X_train = data[:,:-1]
y_train = data[:,-1]
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)

print("Loading test set")
data = np.array(arff.load(open('TestFold00w4.arff', 'r'))["data"])
X_test = data[:,:-1]
y_test = data[:,-1]
y_test = le.transform(y_test)

stop = time.perf_counter()
print("Time spent loading data: {0}".format(stop-start))

start = time.perf_counter()
print('Original dataset shape %s' % Counter(y_train))

rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

print('Resampled dataset shape %s' % Counter(y_train_res))

stop = time.perf_counter()
print("Time spent in the random undersampling: {0}".format(stop-start))

classifier = RandomForestClassifier(n_estimators=250, verbose=10,n_jobs=4)

start = time.perf_counter()

print("Training classifier on the original training set")
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)
score = metrics.f1_score(y_test,pred)
print("F1 score on the original training set: {0}".format(score))

stop = time.perf_counter()
print("Time spent training on the full dataset: {0}".format(stop-start))

start = time.perf_counter()
print("Training classifier on the reduced training set")
classifier.fit(X_train_res,y_train_res)
pred = classifier.predict(X_test)
score = metrics.f1_score(y_test,pred)
print("F1 score on the reduced training set: {0}".format(score))

stop = time.perf_counter()
print("Time spent training on the reduced dataset: {0}".format(stop-start))
