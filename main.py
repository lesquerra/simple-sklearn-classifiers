import sys
import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection, svm, linear_model, neighbors, tree, ensemble

input_data = pd.read_csv(sys.argv[1], delimiter = ",", header = 0)
output_file = str(sys.argv[2])


# Data preparation
nrow = input_data.shape[0]
ncol = input_data.shape[1] - 1

X = input_data.iloc[:,0:ncol]
y = input_data.iloc[:,ncol]


# Split train/test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4, stratify = y, random_state=3944)


# Models

## SVM Linear Kernel
parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}
scores = []

svc = svm.SVC(kernel = 'linear')
clf = model_selection.GridSearchCV(svc, param_grid = parameters, cv = 5, return_train_score=True)
clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_train_score']

model_scores = np.append('svm_linear', np.append(max(scores), clf.score(X_test, y_test)))

## SVM Polynomial Kernel
parameters = {'C':[0.1, 1, 3], 'degree':[4, 5, 6], 'gamma':[0.1, 0.5]}
scores = []

svc = svm.SVC(kernel = 'poly')
clf = model_selection.GridSearchCV(svc, param_grid = parameters, cv = 5, return_train_score=True)
clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_train_score']

model_scores = np.vstack((model_scores, np.append('svm_polynomial', np.append(max(scores), clf.score(X_test, y_test)))))

## SVM RBF Kernel
parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100], 'gamma':[0.1, 0.5, 1, 3, 6, 10]}
scores = []

svc = svm.SVC(kernel = 'rbf')
clf = model_selection.GridSearchCV(svc, param_grid = parameters, cv = 5, return_train_score=True)
clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_train_score']

model_scores = np.vstack((model_scores, np.append('svm_rbf', np.append(max(scores), clf.score(X_test, y_test)))))

## Logistic Regression
parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}
scores = []

lr = linear_model.LogisticRegression()
clf = model_selection.GridSearchCV(lr, param_grid = parameters, cv = 5, return_train_score=True)
clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_train_score']

model_scores = np.vstack((model_scores, np.append('logistic', np.append(max(scores), clf.score(X_test, y_test)))))

## kNN
parameters = {'n_neighbors': np.array(range(50)) + 1, 'leaf_size': (np.array(range(12)) + 1) * 5}

knn = neighbors.KNeighborsClassifier()
clf = model_selection.GridSearchCV(knn, param_grid = parameters, cv = 5, return_train_score=True)
clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_train_score']

model_scores = np.vstack((model_scores, np.append('knn', np.append(max(scores), clf.score(X_test, y_test)))))

## Decision Trees
parameters = {'max_depth': np.array(range(50)) + 1, 'min_samples_split': np.array(range(9)) + 2}

tr = tree.DecisionTreeClassifier()
clf = model_selection.GridSearchCV(tr, param_grid = parameters, cv = 5, return_train_score=True)
clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_train_score']

model_scores = np.vstack((model_scores, np.append('decision_tree', np.append(max(scores), clf.score(X_test, y_test)))))

## Random Forest
parameters = {'max_depth': np.array(range(50)) + 1, 'min_samples_split': np.array(range(9)) + 2}

rf = ensemble.RandomForestClassifier()
clf = model_selection.GridSearchCV(rf, param_grid = parameters, cv = 5, return_train_score=True)
clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_train_score']

model_scores = np.vstack((model_scores, np.append('random_forest', np.append(max(scores), clf.score(X_test, y_test)))))


# Write all outputs to file
np.savetxt(output_file, model_scores, delimiter=",", fmt='%s')
