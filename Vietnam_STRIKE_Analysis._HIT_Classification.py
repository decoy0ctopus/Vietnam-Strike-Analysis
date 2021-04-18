# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:51:43 2020

@author: pater
"""


# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing the dataset

dataset = pd.read_csv('Vietnam_Bombing_Operations_STRIKE_HIT_YN.csv')


dataset['MSNDATE'] = pd.to_datetime(dataset['MSNDATE'], infer_datetime_format=True)
dataset['MSNDATE'] = dataset['MSNDATE'].values.astype(np.int64) / 10**9

 # Get rid of missing data and replace with most frequent

from sklearn.impute import SimpleImputer
        
imputer = SimpleImputer(strategy='most_frequent')
#dataset['WEAPONTYPE'] = dataset['WEAPONTYPE'].fillna("")
dataset['WEAPONTYPE'] = imputer.fit_transform(dataset['WEAPONTYPE'].values.reshape(-1, 1))
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
dataset['WEAPONSLOADEDWEIGHT'] = imputer.fit_transform(dataset['WEAPONSLOADEDWEIGHT'].values.reshape(-1, 1))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['PERIODOFDAY'] = le.fit_transform(dataset['PERIODOFDAY'])

# use pd.concat to join the new columns with your original dataframe
dataset = pd.get_dummies(data=dataset, columns=['MILSERVICE', 'TAKEOFFLOCATION', 'TGTTYPE','WEAPONTYPE', 'AIRCRAFT_ROOT', 'UNIT'])

#Split dependent and independent variables

X = dataset.loc[:, dataset.columns != 'RESULTSBDA'].values
y = dataset.loc[:, dataset.columns == 'RESULTSBDA'].values # 1=HIT 0=MISS

X = np.nan_to_num(X)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Feature Scaling of independent varables

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 0:10] = sc.fit_transform(X_train[:, 0:10])
X_test[:, 0:10] = sc.fit_transform(X_test[:, 0:10])

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_train, y_train.ravel())

y_pred = classifier.predict(X_test)

# Comparing predictions with y_test values

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

#making the confusion matrix and the accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

# Importing Keras libraries

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Initialising the Neural Network
def baseline_classifier():
    
    classifier = Sequential()
    
    classifier.add(Dense(activation='relu', units = 256))
    classifier.add(Dense(activation='relu', units = 256))
    classifier.add(Dense(activation='sigmoid', units = 1))
    
    # Compiling the ANN
    
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

# # Train your model

classifier = KerasClassifier(build_fn=baseline_classifier, verbose=0)
classifier.fit(X_train, y_train, batch_size = 60, epochs = 10)

# use your model to make a prediction on unseen data

y_pred = classifier.predict(X_test)

# Asssess the accuracy of your model and explain your key findings

y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#making the confusion matrix and the accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

import eli5
from eli5.sklearn import PermutationImportance
from eli5.formatters.text import format_as_text

perm = PermutationImportance(classifier).fit(X_test, y_test)
expl = eli5.explain_weights(perm, feature_names = [*dataset.loc[:, dataset.columns != 'RESULTSBDA']])
print(format_as_text(expl, highlight_spaces=None))

# define the grid search parameters

from sklearn.model_selection import GridSearchCV
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

