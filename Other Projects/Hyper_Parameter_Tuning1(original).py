# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:28:58 2021

@author: pc
"""
import pandas as pd


#Import 'GridSearchCV' and 'make_scorer'
#from Main import *
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

#import libraries
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #train test split library 
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from numpy import mean
from numpy import std

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

df= pd.read_csv('preprocessed_data(final).csv', encoding= 'unicode_escape') 
df = df.drop(['Unnamed: 0'], axis = 1)

print (df.head)



X3 = df.drop(['FTR'], axis=1)
X3 = X3.astype('int64')
X3 = X3.values


Y3 = df['FTR']
Y3 = Y3.values


x3train, x3test, y3train, y3test = train_test_split(X3,Y3, test_size = 0.2, random_state=1)

# X3['FTR'].value_counts()


clf = RandomForestClassifier() 

#Fitting the data
clf.fit(x3train, y3train)
# Now get the score using score method
print('The score on FTR TEST DATA') 
print(clf.score(x3test, y3test))

y_pred = clf.predict(x3test)
     # getting the classfication report
print(classification_report(y3test, y_pred))
from sklearn.metrics import confusion_matrix as cm
cm = cm(y3test, y_pred)
# potting graphical confusion matrix on the test data 

import matplotlib.pyplot as plt  
import seaborn as sns
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = "g"); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
plt.show()

#with cross validation spliting
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

models = []
models.append(('RF', RandomForestClassifier()))
#models.append(('SVM', SVC()))

results = []
resultsavg = []
names = []
resultsstd = []


#leave this here.
####with test train split the dataset is splited randomly 80,20 or 70,30. so when randomly splited. we get fluctuated accuracies. everytime.:
#i understand 70,30 + 80,20 how is cross validation done. cross validation used with 90,10 splition. it split the data 10 times. it called fold. so we used 10 k fold. where no of folds are 10. 
##when working with 1st fold. data is splited 90 to train 10 test. similarly in next fold. different data is goes to train 90%, 10% to test. in the end we get 10 accuraies. so take mean of these 10 accuracies. the mean of these 10 accuracies is our final accuracy.. 
## so our results not fluctues much iwht cross valiation. k is the number . we can use what ever value for number of fold. but usualy 10 is been used. got it? 
#is there any other purposes . it gives accuracte accuracy. okay perfect will research more thanks for this



for name, model in models:
    cv_results = cross_val_score(model, X3, Y3, cv=kfold, scoring='accuracy')
    
    results.append(cv_results)
    resultsavg.append(cv_results.mean())
    resultsstd.append(cv_results.std())
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# GRID search with cross validation
# configure the cross-validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X3):
    # split data
    X_train, X_test = X3[train_ix, :], X3[test_ix, :]
    y_train, y_test = Y3[train_ix], Y3[test_ix]
    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
    # define the model
    model = RandomForestClassifier(random_state=1)
    # define search space
    space = dict()
    space['n_estimators'] = [100, 120, 150]
    space['min_samples_split'] = [2, 4]
    space['min_samples_leaf'] = [1, 2]
    # define search
    search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
    # execute search
    result = search.fit(X_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    # evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
    # evaluate the model
    acc = accuracy_score(y_test, yhat)
    # store the result
    outer_results.append(acc)
    # report progress
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))


#GRID search with test train split.
param_comb = 5
folds = 3


n_estimators = [100,120, 150]
#max_features = [2, 4,6]
#max_depth = np.arange(1,10)
#n_estimators = [int(x) for x in np.linspace(start=10, stop=80, num=10)]
#max_features = ['auto', 'sqrt']
min_samples_split = [2, 4]
min_samples_leaf = [1, 2]
#max_depth = [2,4]
#min_samples_leaf = [1, 2, 4]
#min_samples_split = [1,2]
#bootstrap = [True, False]


params={
        'n_estimators': n_estimators,
               #'max_features': max_features,
               #'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               #'min_samples_split': min_samples_split,
               #'min_samples_leaf': min_samples_leaf,
               #'bootstrap': bootstrap
               }


#skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

#xgb = RandomForestClassifier(random_state=42) 
xgb = RandomForestClassifier() 
#xgb = XGBClassifier(use_label_encoder = False )
#random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, n_jobs = -1, cv=skf.split(X3,Y3), verbose=3, random_state=1001 )
#grid_cv = GridSearchCV(xgb, param_grid=params, n_jobs = -1, cv=skf.split(X3,Y3), verbose=3)
grid_cv = GridSearchCV(xgb, param_grid=params, cv=5, verbose=2, n_jobs=4)

#random_search.fit(X3, Y3)
grid_cv.fit(X3, Y3)

print(grid_cv.best_estimator_)
model = grid_cv.best_estimator_


model.fit(x3train, y3train)
pred = model.predict(x3test)

print(classification_report(y3test, pred))


print('The score on FTR TEST DATA') 
print(model.score(x3test, y3test))



