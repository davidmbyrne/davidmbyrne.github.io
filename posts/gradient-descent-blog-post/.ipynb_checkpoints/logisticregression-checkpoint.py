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
        
    def sigmoid(self, z):
        return 1 / (1+ np.exp(-z))
    
    def logistic_loss(self, y_hat, y):
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))
    
    def empirical_risk(self, X, y, loss, w):
        y_hat = self.predict(X, w)
        return self.logistic_loss(y_hat, y).mean()
    
    def gradient(self, w, X, y):
        w = w.reshape(self.X_.shape[1] , 1)
        sigdot = (np.dot(X, w) >= 0).astype(int)
        return np.sum(np.multiply(X,(self.sigmoid(sigdot) - y.reshape(len(y),1))))/X.shape[0]
    
    def fit_stochastic(self, X, y, m_epochs, alpha, batch_size):
        prev_loss = np.inf
        self.loss_history = []
        self.score_history = []
        w = np.random.rand(X.shape[1], 1)
        bias = np.random.uniform(0,1)
        self.w_ = np.append(w, -bias)
        self.X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        n = X.shape[0]
        for j in np.arange(m_epochs):
            
            order = np.arange(n)
            np.random.shuffle(order)

            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = self.X_[batch,:]
                y_batch = y[batch]
                grad = self.gradient(self.w_, x_batch, y_batch) 
                self.w_ -= alpha*grad                      
            new_loss = self.empirical_risk(self.X_, y, self.logistic_loss, self.w_)
            self.loss_history.append(new_loss)
            self.currentscore = self.score(self.X_, y)
            self.score_history.append(self.currentscore)
    
            if np.isclose(new_loss, prev_loss):
                self.stop = j
                return
            else:
                prev_loss = new_loss
        print("Too many epochs")
                
    def fit(self, X, y, m_epochs, alpha):
        prev_loss = np.inf
        self.loss_history = []
        self.score_history = []
        w = np.random.rand(X.shape[1], 1)
        bias = np.random.uniform(0,1)
        self.w_ = np.append(w, -bias)
        self.X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        for j in np.arange(m_epochs):
            grad = self.gradient(self.w_, self.X_, y)
            self.w_ -= alpha*grad
            new_loss = self.empirical_risk(self.X_, y, self.logistic_loss(self.predict(self.X_, self.w_), y), self.w_)
            self.loss_history.append(self.empirical_risk(self.X_, y, self.logistic_loss, self.w_))
            self.score_history.append(self.score(self.X_, y))
            
            if np.isclose(new_loss, prev_loss): 
                self.stop = j
                return
            else:
                prev_loss = new_loss
        self.stop = "too many"