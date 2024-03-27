from itertools import combinations
import numpy as np
import scipy
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')



class LogisticRegressionIWLS:
    
    def __init__(self, delta = 1, LL = 0,interact = False):
        self.delta = delta
        self.LL = LL
        self.beta = None
        self.intract = interact 


    def interact(self, X):
        # Interact between variables - add columns
        new_cols ={}
        for col in combinations(X.columns, 2):
            new_cols[f"{col[0]}_{col[1]}"] = X[col[0]] * X[col[1]]        
        new_df = pd.DataFrame(new_cols)
        result = pd.concat([X, new_df], axis=1)
        return result

    def logit(self, x):
        if isinstance(x, list):
            x = np.array(x)
        # logit function
        return np.log(x / (1 - x))
        
    def logistic(self, x):
        if isinstance(x, list):
            x = np.array(x)
        # sigmoid function
        return 1/(1+np.exp(-x))
        
    def var(self, x):
        if isinstance(x, list):
            x = np.array(x)
        # variance
        return x*(1-x)
    
    def logit_prime(self, x):
        if isinstance(x, list):
            x = np.array(x)
        # derivative of the logistic sigmoid function
        return 1/(x*(1-x))

    def WLS(self, X, W, Z):
        # Weighted Least Squares
        return scipy.linalg.inv(np.array(X).T.dot(W).dot(np.array(X))).dot(np.array(X).T.dot(W).dot(Z))

    def ll(self, p, y):
        # Log-likelihood function
        return sum(y*np.log(p)+(np.ones(len(y))-y)*np.log(np.ones(len(y))-p))
    
    def fit(self,X,y):
        if self.interact:
            X_new = self.interact(X)
        else:
            X_new = X
        mu = [0.5]*len(y)
        i = 1
        # Iterate
        while self.delta> 0.00001 :
            Z = self.logit(mu) + (y-mu)*self.logit_prime(mu)
            W =  np.diag((1 / (self.logit_prime(mu)**2 * self.var(mu))))
            self.beta = self.WLS(X = X_new, W = W, Z = Z)
            eta = np.array(X_new @ self.beta)
            mu = self.logistic(eta)
            LL_old = self.LL
            self.LL = self.ll(p = mu, y = y)
            self.delta = np.abs(self.LL - LL_old)
            i += 1 
        return self.beta    
    
    def predict(self, X):
        if self.interact:
            X_new = self.interact(X)
        else:
            X_new = X
        return [1 if el > 0.5 else 0 for el in self.logistic(X_new @ self.beta)]
    