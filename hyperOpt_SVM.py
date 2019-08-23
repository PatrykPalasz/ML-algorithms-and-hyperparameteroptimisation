from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

import numpy as np
import datetime

x_train = []
x_test = []
y_train = []
y_test = []
max = 0

from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from utils import normalize

def getSpace():
  space = { 'penality' : hp.choice('penality', [0.01, 0.1, 1, 10, 50, 100, 150, 200, 300, 400, 500, 1000]),
            'kernel'   : hp.choice('kernel', [{'function': 'linear'}, 
                                              {'function': 'poly',    'gamma1' : hp.uniform('gamma1', 0.0, 1.0),  'coef01' : hp.uniform('coef01', 0.0, 5.0), 'degree' : hp.choice('degree', np.arange(1, 6, dtype=int))}, # polynomial degree, used only if 'poly'
                                              {'function': 'rbf',     'gamma3' : hp.uniform('gamma3', 0.0, 1.0)} # gamma used for poly, rbf, sigmoid (if auto then 1/n_features will be used)
                                             ]), 
            'shrinking' : hp.choice('shrinking', [True, False])
        }
  return space

def getModel(params):
  cache = 700000
  iter = 20000

  if params['kernel']['function'] == 'linear':
    model = SVC(C=params['penality'], cache_size=cache, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel=params['kernel']['function'],
      max_iter=iter, probability=False, random_state=None, shrinking=params['shrinking'],
      tol=0.0001, verbose=False)
    return model

  if params['kernel']['function'] == 'rbf':                                                                               
    model = SVC(C=params['penality'], cache_size=cache, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=params['kernel']['gamma3'], kernel=params['kernel']['function'],
      max_iter=iter, probability=False, random_state=None, shrinking=params['shrinking'],
      tol=0.0001, verbose=False)
    return model

  if params['kernel']['function'] == 'sigmoid':
    model = SVC(C=params['penality'], cache_size=cache, class_weight=None, coef0=params['kernel']['coef02'],
      decision_function_shape='ovr', degree=3, gamma=params['kernel']['gamma2'], kernel=params['kernel']['function'],
      max_iter=iter, probability=False, random_state=None, shrinking=params['shrinking'],
      tol=0.0001, verbose=False)
    return model

  if params['kernel']['function'] == 'poly':
    model = SVC(C=params['penality'], cache_size=cache, class_weight=None, coef0=params['kernel']['coef01'],
      decision_function_shape='ovr', degree=params['kernel']['degree'], gamma=params['kernel']['gamma1'], kernel=params['kernel']['function'],
      max_iter=iter, probability=False, random_state=None, shrinking=params['shrinking'],
      tol=0.0001, verbose=False)
    return model
  
  print('You should not see that info - check getSpace!')

def objective(params):   
    global max
    
    startTime = datetime.datetime.now()
    print(f'{startTime}: Params testing: {params}')
    model = getModel(params)
    model.fit(x_train, y_train, ) 
    finishTime = datetime.datetime.now()
    time = (finishTime - startTime).total_seconds();
    y_pred = model.predict(x_test)

    auc = roc_auc_score(y_test, y_pred)
    if(auc > max):
      max = auc
      print(f'MAX: {max}')
    print(f'AUC:{acc} time: {time}s max: {max}')
    return {'loss': -acc, 'status': STATUS_OK}

def doSVMHyperOpt(x, y):
  global x_train
  global x_test
  global y_train
  global y_test
  global max

  max = 0
  x = normalize(x)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
  
  trials = Trials()
  best = fmin(objective, space=getSpace(), algo=tpe.suggest, max_evals = 100, trials=trials)
  print(f'BEST: {best}')
  print('Trials: {trials}')

