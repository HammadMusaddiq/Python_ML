# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:28:58 2021

@author: pc
"""

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
#from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt  
import seaborn as sns

from numpy import mean
from numpy import std

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df= pd.read_csv('HyperTuning_preprocessed_data(final)1.csv', encoding= 'unicode_escape') 
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
cm = confusion_matrix(y3test, y_pred)

# potting graphical confusion matrix on the test data 
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = "g"); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
plt.show()


#leave this here.
####with test train split the dataset is splited randomly 80,20 or 70,30. so when randomly splited. we get fluctuated accuracies. everytime.:
#i understand 70,30 + 80,20 how is cross validation done. cross validation used with 90,10 splition. it split the data 10 times. it called fold. so we used 10 k fold. where no of folds are 10. 
##when working with 1st fold. data is splited 90 to train 10 test. similarly in next fold. different data is goes to train 90%, 10% to test. in the end we get 10 accuraies. so take mean of these 10 accuracies. the mean of these 10 accuracies is our final accuracy.. 
## so our results not fluctues much iwht cross valiation. k is the number . we can use what ever value for number of fold. but usualy 10 is been used. got it? 
#is there any other purposes . it gives accuracte accuracy. okay perfect will research more thanks for this



# GRID search with 10 k-fold cross validation
print("Hyper Parameter Tuning with Folds")
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X3):
    # split data
    X_train, X_test = X3[train_ix, :], X3[test_ix, :]
    y_train, y_test = Y3[train_ix], Y3[test_ix]
    # configure the cross-validation procedure of Grid Search
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
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

print(classification_report(y_test, yhat))

print('The score on FTR TEST DATA with Hyper Tuning') 
# summarize the estimated performance of the model
print(mean(outer_results))

#only 3 decimal points of accuracy
outer_results1 = np.around(outer_results,3)

#data frame of ploting
df_plot = pd.DataFrame({'Fold' : np.arange(1,11)})
df_plot['Accuracy'] = outer_results1

#for Mean Line
x = [1,10,-1]
y_mean = [mean(outer_results)]*len(x)

#ploting
clrs = ['blue' if (x < max(df['Accuracy'])) else 'red' for x in df['Accuracy'] ]
ax = df.plot.bar(x='Fold', y='Accuracy', rot=0, color=clrs)
ax.plot(x,y_mean, label='Mean', linestyle='--', color = 'black')
ax.set_ylim(0.80,0.925)
ax.set_ylabel('Accuracy')
ax.set_title('Hyperparameter Tuning with 10 Fold')
legend = ax.legend(loc='upper right')


