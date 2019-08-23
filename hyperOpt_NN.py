from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

import numpy as np

x_train = []
x_test = []
y_train = []
y_test = []

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from utils import normalize

def getSpace():
  space = { 'num_layers' : hp.choice('num_layers', np.arange(1, 20, dtype=int)),
            'units'      : hp.choice('units', np.arange(32, 124, dtype=int)),
            'batch_size' : hp.choice('batch_size', np.arange(32, 256, dtype=int)),
            'nb_epochs'  : hp.choice('epochs', np.arange(100, 200, dtype=int)),
            'optimizer'  : hp.choice('optimizer', ['adadelta','adam','rmsprop', 'SGD', 'adagrad', 'adamax', 'nadam']), 
            'activation' : hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'hard_sigmoid', 'softsign', 'softplus', 'softmax', 'elu', 'selu'])
        }
  return space

def getModel(params):
  model = Sequential()
  model.add(Dense(params['units'], activation = params['activation'], input_dim = len(x_train[0])))
  
  for i  in range(0, params['num_layers']):
    model.add(Dense(params['units'], activation = params['activation']))

  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  return model

def objective(params):   
    print ('Params testing: ', params)
    model = getModel(params)
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])

    model.fit(x_train, y_train, epochs=params['nb_epochs'], batch_size=params['batch_size'], verbose = 0)

    pred_auc = model.predict_proba(x_test, batch_size = 128, verbose = 0)
    auc = roc_auc_score(y_test, pred_auc)
    print('AUC:', auc)
    return {'loss': -auc, 'status': STATUS_OK}

def doNNHyperOpt(x, y):
  global x_train
  global x_test
  global y_train
  global y_test

  x = normalize(x)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
  
  trials = Trials()
  best = fmin(objective, space=getSpace(), algo=tpe.suggest, max_evals = 200, trials=trials)
  print('BEST: ')
  print(best)
