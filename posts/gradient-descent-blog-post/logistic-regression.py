import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

class LogisticRegression:
    
    def __init__(self):
        pass
    
    def score(self, X, y):
        return np.sum((y == np.transpose(self.guess)))/(X.shape[0])
    
    def predict(self, X, w):
        self.guess = (np.dot(X, w) >= 0).astype(int)
        return self.guess
        
    def fit(self, X, y):
        self.loss_history = []
        self.score_history = []
        w = np.random.rand(2,1)
        bias = np.random.uniform(0,1)
        self.w_ = np.append(w, -bias)
        self.X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
    def sigmoid(self, z):
        return 1 / (1+ np.exp(-z))
    
    def logistic_loss(self, y_hat, y):
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))
    
    def empirical_risk(self, X, y, loss, w):
        y_hat = self.predict(X, w)
        return loss(y_hat, y).mean()
    
    def gradient(self, w, X, y):
        w = w.reshape(3,1)
        sigdot = (np.dot(X, w) >= 0).astype(int)
        return np.sum(np.multiply(X,(self.sigmoid(sigdot) - y.reshape(100,1))))