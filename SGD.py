from itertools import combinations
import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')


class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=1, interaction = False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.interaction = interaction

    def interact(self, X):
        # Interact between variables - add columns
        new_cols ={}
        for col in combinations(X.columns, 2):
            new_cols[f"{col[0]}_{col[1]}"] = X[col[0]] * X[col[1]]        
        new_df = pd.DataFrame(new_cols)
        result = pd.concat([X, new_df], axis=1)
        return result
    
    def logistic(self, x):
        if isinstance(x, list):
            x = np.array(x)
        # Sigmoid function
        return 1/(1+np.exp(-x))

    def fit(self, X, y):
        # Interact between variables
        if self.interaction:
            X_new = self.interact(X)
        else:
            X_new = X
        num_samples, num_features = X_new.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.epochs):
            # Shuffle the data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_new.iloc[indices,:]
            y_shuffled = y.iloc[indices]

            # Find batches
            for i in range(0, num_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                # Compute predictions
                y_pred = self.logistic(np.dot(X_batch, self.weights) + self.bias)

                # Compute gradients
                dw = (1/self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (1/self.batch_size) * np.sum(y_pred - y_batch)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        if self.interaction:
            X_new = self.interact(X)
        else:
            X_new = X
        y_pred = self.logistic(np.dot(X_new, self.weights) + self.bias)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class
