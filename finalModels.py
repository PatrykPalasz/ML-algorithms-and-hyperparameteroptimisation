from __future__ import print_function
import sys
import os
import argparse
import datetime

from keras.models import Sequential
from keras.layers import Dense, Dropout   
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier

def getOptimizedNNModel(x_train, y_train):
  startTime = datetime.datetime.now()

  classifier = Sequential()
  classifier.add(Dense(112, activation = 'softsign', input_dim = len(x_train[0])))
  classifier.add(Dense(112, activation = 'softsign'))
  classifier.add(Dense(112, activation = 'softsign'))
  classifier.add(Dense(112, activation = 'softsign'))
  classifier.add(Dense(1 ,  activation = 'sigmoid'))

  classifier.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])
  history = classifier.fit(x_train, y_train, batch_size=158, epochs=123, verbose=0) 

  finishTime = datetime.datetime.now()
  print("Model NN:     {0}".format(finishTime - startTime)) 
  return classifier, history

def getOptimizedSVMModel(x_train, y_train):
  startTime = datetime.datetime.now()

  model = SVC(C=50, cache_size=700, class_weight='balanced', coef0=0.0,                      
    decision_function_shape='ovr', degree=3, gamma=0.1717011742129446958, kernel='rbf',
    max_iter=40000, probability=False, random_state=np.random.RandomState(0), shrinking=True,          
    tol=0.0001, verbose=False)

  model.fit(x_train, y_train)
  finishTime = datetime.datetime.now()
  print("Model SVM:    {0}".format(finishTime - startTime)) 
  return model

def getOptimizedBDTModel(x_train, y_train):
  startTime = datetime.datetime.now()

  model = BaggingClassifier(n_estimators=20, max_samples=0.95)
  model.fit(x_train, y_train)

  finishTime = datetime.datetime.now()
  print("Model BT:     {0}".format(finishTime - startTime)) 
  return model

