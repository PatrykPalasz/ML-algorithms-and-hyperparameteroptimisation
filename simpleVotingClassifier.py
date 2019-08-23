import numpy as np 

class simpleVotingClassifier(object):
    """ Implements a voting classifier for pre-trained classifiers"""
    def __init__(self, estimators, weights = None):
        self.estimators = estimators
        self.weights = weights

    def predict(self, X):
        # get values
        self.predictions = np.zeros([X.shape[0], len(self.estimators)])
        if(self.weights == None): 
          self.weights = np.ones(len(self.estimators));
        for i, clf in enumerate(self.estimators):
          pred = np.squeeze(clf.predict(X));
          if(len(pred.shape) > 1):
            self.predictions[:, i] = pred[:,1]
          else:
            self.predictions[:, i] = pred;
        # apply voting 
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
          y[i] = np.average(self.predictions[i,:], weights=self.weights)
        return y