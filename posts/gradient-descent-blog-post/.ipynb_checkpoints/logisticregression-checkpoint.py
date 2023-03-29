import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

class LogisticRegression:
    '''
    Initialize the logistic regression object
    '''
    def __init__(self):
        pass
    
    '''
    Scores the current model by finding how many points in our guess vector (y_hat)
    match the actual observation vector y
    '''
    def score(self, X, y):
        return np.sum((y == np.transpose(self.guess)))/(X.shape[0])
    
    '''
    Utilizes the dot product to determine the classification of each point
    based on the sign of the value
    '''
    def predict(self, X, w):
        self.guess = (np.dot(X, w) >= 0).astype(int)
        return self.guess
        
    '''
    Sigmoid function
    '''
    def sigmoid(self, z):
        return 1 / (1+ np.exp(-z))
    
    '''
    Implementation of the logistic loss function using the sigmoid function
    '''
    def logistic_loss(self, y_hat, y):
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))
    
    '''
    Creates a vector of predictions y_hat and calculates empirical risk using the logistic
    loss function
    '''
    def empirical_risk(self, X, y, loss, w):
        y_hat = self.predict(X, w)
        return self.logistic_loss(y_hat, y).mean()
    
    '''
    
    '''
    def gradient(self, w, X, y):
        w = w.reshape(self.X_.shape[1] , 1)
        sigdot = (np.dot(X, w) >= 0).astype(int)
        return np.sum(np.multiply(X,(self.sigmoid(sigdot) - y.reshape(len(y),1))))/X.shape[0]
    
    '''
    
    '''
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
            
            self.stop = j
            if np.isclose(new_loss, prev_loss):
                break
            else:
                prev_loss = new_loss
        if self.stop < m_epochs-1:
            print("Converged after " + str(self.stop) + " tries")
        else:
            print("Reached maximum epochs before converging")
    
    '''
    
    '''
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
            self.loss_history.append(new_loss)
            self.score_history.append(self.score(self.X_, y))
            
            self.stop = j
            if np.isclose(new_loss, prev_loss):
                break
            else:
                prev_loss = new_loss
        if self.stop < m_epochs-1:
            print("Converged after " + str(self.stop) + " tries")
        else:
            print("Reached maximum epochs before converging")
            
        