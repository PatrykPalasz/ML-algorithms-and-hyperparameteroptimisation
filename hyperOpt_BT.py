from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

import numpy as np

x_train = []
x_test = []
y_train = []
y_test = []

from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from utils import normalize

def getSpace():
  space = { 'estimators' : hp.choice('estimators', np.arange(3, 30, dtype=int)),
            'features'   : hp.uniform('fatures', 0.0, 1.0),
            'samples'    : hp.uniform('samples', 0.0, 1.0)
        }
  return space

def getModel(params):
  model = BaggingClassifier(base_estimator=None,  # decision tree
                            n_estimators=params['estimators'],
                            max_features=params['features'],
                            max_samples=params['samples'],
                            n_jobs=-1
                            )
  return model

def objective(params):   
    print ('Params testing: ', params)
    model = getModel(params)
    model.fit(x_train, y_train) 
    y_pred = model.predict(x_test)

    auc = roc_auc_score(y_test, y_pred)
    print('AUC:', auc)
    return {'loss': -auc, 'status': STATUS_OK}

def doBTHyperOpt(x, y):
  global x_train
  global x_test
  global y_train
  global y_test

  x = normalize(x)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
  
  trials = Trials()
  best = fmin(objective, space=getSpace(), algo=tpe.suggest, max_evals = 30, trials=trials)
  print('BEST: ')
  print(best)
