#!/usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle5 as pickle
import pandas as pd

#Load the dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=75, test_size=0.75)

#Compile our Support Vector Classifier model with training. 
model_svc = svm.SVC(gamma='auto')
model_svc.fit(X_train, y_train)

#Get accuracy score of the SVM model.
accuracy_svc = model_svc.score(X_test, y_test)
print("Accuracy of Support Vector Classifier is: {}".format(round(accuracy_svc, 3)))

#Compile the Random Forest Classifier model with training.
model_rfc = RandomForestClassifier(n_estimators=25)
model_rfc.fit(X_train, y_train)

#Get accurcy score of the RFC model
accuracy_rfc = model_rfc.score(X_test, y_test)
print("Accuracy of Random Forest Classifier is: {}".format(round(accuracy_rfc, 3)))

with open('model_svc.pkl', 'wb') as model_svc_pickle:
    pickle.dump(model_svc, model_svc_pickle)

with open('model_rfc.pkl', 'wb') as model_rfc_pickle:
    pickle.dump(model_rfc, model_rfc_pickle)