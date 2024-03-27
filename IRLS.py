from itertools import combinations
import numpy as np
import scipy
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
from sklearn.metrics import balanced_accuracy_score

class LogisticRegressionIWLS:
    
    def __init__(self, epoch = 1000, interact = False,  early_stopping=True, patience=5, LL = 0, delta = 1):
        self.delta = delta
        self.LL = LL
        self.beta = None
        self.interact = interact 
        self.epoch = epoch
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0
        self.predicts = []
        self.deltas = []
        self.val_preds = []



    def interaction(self, X):
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
        
    def sigmoid(self, x):
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
    
    def train(self,X,y, X_val=None, y_val=None):
        if self.interact:
            X_new = self.interaction(X)
            # if X_val is not None:
            #     X_val_new = self.interaction(X_val)
        else:
            X_new = X
            # X_val_new = X_val

        mu = [0.5]*len(y)
        i = 1
        # Iterate
        # https://www.cs.toronto.edu/~duvenaud/papers/md_paper.pdf
        while  i < 100: #self.delta> 0 and
        # for e in range(self.epoch):
            Z = self.logit(mu) + (y-mu)*self.logit_prime(mu)
            W =  np.diag((1 / (self.logit_prime(mu)**2 * self.var(mu))))
            self.beta = self.WLS(X = X_new, W = W, Z = Z)
            eta = np.array(X_new @ self.beta)
            mu = self.sigmoid(eta)
            print(mu)
            LL_old = self.LL
            self.LL = self.ll(p = mu, y = y)
            self.delta = np.abs(self.LL - LL_old)
            self.deltas.append(self.delta)
            self.predicts.append(balanced_accuracy_score([1 if el > 0.5 else 0 for el in mu], y))
            self.val_preds.append(balanced_accuracy_score([1 if el > 0.5 else 0 for el in self.predict(X_val)], y_val))

            i += 1 
            # if self.early_stopping and X_val is not None and y_val is not None:
            #     val_loss = self.compute_loss(X_val, y_val)
            #     if val_loss < self.best_loss:
            #         self.best_loss = val_loss
            #         self.counter = 0
            #     else:
            #         self.counter += 1
            #         if self.counter >= self.patience:
            #             print(f"Stopping early at epoch {e+1} as validation loss has not improved for {self.patience} consecutive epochs.")
            #             break
        return self.beta    
    
    def predict(self, X):
        return [1 if el > 0.5 else 0 for el in self.predict_proba(X)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return balanced_accuracy_score(predictions, y)
    
    def predict_proba(self, X):
        if self.interact:
            X_new = self.interaction(X)
        else:
            X_new = X
        return  self.sigmoid(X_new @ self.beta)      


    def compute_loss(self, X, y):
        predictions = self.predict(X)
        # print(np.mean(predictions == y))
        loss = -np.mean(y * np.log(predictions) + (np.ones(len(y)) - y) * np.log(np.ones(len(y)) - predictions))
        return loss

    