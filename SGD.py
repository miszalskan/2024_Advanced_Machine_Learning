from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import warnings 
warnings.filterwarnings('ignore')


class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=1, interact = False, delta = 1, LL = 0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.interact = interact

        self.delta = delta
        self.LL = LL
   
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
    
    def sigmoid(self, x):
        if isinstance(x, list):
            x = np.array(x)
        # Sigmoid function
        return 1/(1+np.exp(-x))
    
    def ll(self, p, y):
        # Log-likelihood function
        return sum(y*np.log(p)+(np.ones(len(y))-y)*np.log(np.ones(len(y))-p))
    
    def fit(self, X, y, X_val, y_val):
        # Interact between variables
        if self.interact:
            X_new = self.interaction(X)
        else:
            X_new = X
        num_samples, num_features = X_new.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        i = 0

        # for _ in range(self.epochs):
        while self.delta> 0 and i < 500:

            # Shuffle the data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_new.iloc[indices,:]
            y_shuffled = y.iloc[indices]

            # Find batches
            for i in range(0, num_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                # Compute predictions
                y_pred = self.sigmoid(np.dot(X_batch, self.weights) + self.bias)

                # Compute gradients
                dw = (1/self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (1/self.batch_size) * np.sum(y_pred - y_batch)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            LL_old = self.LL
            p =self.sigmoid(np.dot(X_shuffled, self.weights) + self.bias)
            print(np.dot(X_shuffled, self.weights) + self.bias)
            self.LL = self.ll(p = p, y = y_shuffled)
            print(self.LL)
            self.delta = np.abs(self.LL - LL_old)
            self.deltas.append(self.delta)
            self.predicts.append(balanced_accuracy_score([1 if el > 0.5 else 0 for el in p], y))
            self.val_preds.append(balanced_accuracy_score([1 if el > 0.5 else 0 for el in self.predict(X_val)], y_val))

            i += 1 

    def predict(self, X, inter = False):
        if self.interact and not inter:
            X_new = self.interaction(X)
        else:
            X_new = X
        y_pred = self.sigmoid(np.dot(X_new, self.weights) + self.bias)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class
    
    def score(self, X, y):
        predictions = self.predict(X)
        return balanced_accuracy_score(predictions, y)
    
