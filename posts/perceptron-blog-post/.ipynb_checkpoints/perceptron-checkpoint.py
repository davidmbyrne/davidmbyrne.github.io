import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

class Perceptron:
    def __init__(self):
        pass
    
    def score(self, X, y):
        return np.sum((y == np.transpose(self.guess)))/(X.shape[0])
    
    def predict(self, X):
        self.guess = (np.dot(self.X_, self.w_) >= 0).astype(int)

    '''This is what it does'''
    def fit(self, X, y):
        self.history = []
        self.X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        self.w = np.random.rand(self.X_.shape[1]-1,1)
        bias = np.random.uniform(0, 1)
        self.w_ = np.append(self.w, -bias)
        self.w_ = self.w_.reshape(3,1)
    
    def perceptron_update(self, X, y, w):
        self.w_ = self.w_[:,0].reshape(self.X_.shape[1],1)
        for i in range(self.X_.shape[0]-1):
            val = np.random.randint(0, X.shape[0]-1)
            instance = self.X_[val]
            if np.dot(instance, self.w_) < 0:
                self.w_ = self.w_ + ((2*y[val]-1)*self.X_[val]).reshape(self.X_.shape[1],1)
                self.w_ = self.w_/np.linalg.norm(self.w_)
                break