# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 17:32:37 2020

@author: Robert Dosa
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier


# Data preprocessing

print('Data preprocessing...')
dataset = pd.read_csv('wo_text.csv')
dataset = dataset.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
X = np.array(dataset.drop(['target'], axis=1))
y = np.array(dataset['target'])

# Scaling

sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)


# Building the ANN

def build_cf(optimizer):
    classifier = Sequential()
    classifier.add(Dense(110, kernel_initializer='uniform', activation='relu',
                         input_dim=224))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(110, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(1, kernel_initializer='uniform',
                         activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy',
                       metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_cf)

# Grid search

parameters = {'batch_size': [10, 32],
              'optimizer': ['adam', 'rmsprop'],
              'epochs': [100]}

grid = GridSearchCV(estimator=classifier, param_grid=parameters,
                    scoring='accuracy')

# grid.fit(X_train, y_train)

# print(grid.best_params_)
# print(grid.best_estimator_)


# New classifier

print('Initializing neural network...')
classifier = Sequential()
classifier.add(Dense(250, kernel_initializer='uniform', activation='relu',
                     input_dim=224))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(250, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(1, kernel_initializer='uniform',
                     activation='sigmoid'))
classifier.compile(optimizer='rmsprop', loss='binary_crossentropy',
                   metrics=['accuracy'])
print('Fitting the data...')
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

history = classifier.fit(X_train, y_train, validation_split=0.25, epochs=100,
                         batch_size=10)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
