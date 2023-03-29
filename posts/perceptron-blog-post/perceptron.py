import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

class Perceptron:
    
    """
    Initialize our perceptron object
    """
    def __init__(self):
        pass
    
    
    """
    Computes the score of our algorithm by checking if our guesses (y_hat) and the true
    values for y are equal, summing the resulting vector, and diviiding my the size of our
    matrix X.
    """
    def score(self, X, y):
        return np.sum((y == np.transpose(self.guess)))/(X.shape[0])
    
    
    """
    Computes the dot product of of X and w, assigns a value of 1 if the value is greater 
    than or equal to 0, and 0 otherwise
    """
    def predict(self, X):
        self.guess = (np.dot(self.X_, self.w_) >= 0).astype(int)

        
    '''
    The fit function is the main method of algorithm, initializing our history vector,
    creating our initial weights, as well as preparing our variables for computational 
    use by appending a column of ones to our matrix X and reshaping our weight vector w.
    '''
    def fit(self, X, y, max_iter):
        self.history = []
        self.X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        self.w = np.random.rand(self.X_.shape[1]-1,1)
        bias = np.random.uniform(0, 1)
        self.w_ = np.append(self.w, -bias)
        self.w_ = self.w_.reshape(self.X_.shape[1],1)
        
        for i in range(max_iter):
            self.predict(self.X_) #Make predictions
            self.currscore = self.score(self.X_, y) #Score current weight 
            self.history.append(self.currscore)
            if self.currscore == 1:
                break
            elif self.currscore < 1:
                self.perceptron_update(self.X_, y, self.w_) #Update weight vector if score is not 1.0
    
    
    """
    The perceptron update :
        1) Choose a random point in X
        2) Compute the dot product of the given point the current weight vector w
        3) If the dot product is less than 0 update w as follows:
            w^(t+1) = w^t + (2*y_i-1) * X_i
    """
    def perceptron_update(self, X, y, w):
        self.w_ = self.w_[:,0].reshape(self.X_.shape[1],1)
        #self.w_ = self.w_[:,0].reshape(3,1)
        for i in range(self.X_.shape[0]-1):
            val = np.random.randint(0, X.shape[0]-1)
            instance = self.X_[val]
            if np.dot(instance, self.w_) < 0:
                self.w_ = self.w_ + ((2*y[val]-1)*self.X_[val]).reshape(self.X_.shape[1],1)
                self.w_ = self.w_/np.linalg.norm(self.w_)
                break