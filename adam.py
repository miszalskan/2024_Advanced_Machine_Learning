import numpy as np
from itertools import combinations
import pandas as pd

class LogisticRegressionAdam:
    def __init__(self, learning_rate=0.02, beta1=0.9, beta2=0.999, epochs = 100, epsilon=1e-8, interact = False, early_stopping=True, patience=5):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epochs = epochs
        self.epsilon = epsilon
        self.weights = None 
        self.bias = None 
        self.m_weights = None
        self.v_weights = None
        self.m_bias = None
        self.v_bias = None
        self.t = 0
        self.interact = interact
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0
    
    def interaction(self, X):
        # Interact between variables - add columns
        new_cols ={}
        for col in combinations(X.columns, 2):
            new_cols[f"{col[0]}_{col[1]}"] = X[col[0]] * X[col[1]]        
        new_df = pd.DataFrame(new_cols)
        result = pd.concat([X, new_df], axis=1)
        return result
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def initialize_moments(self):
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = 1e-8
        self.v_bias = 1e-8

    def update_params(self):

        if self.m_weights is None or self.v_weights is None:
            self.initialize_moments()

        self.t += 1

        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * self.grad_w
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * self.grad_b

        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * np.square(self.grad_w)
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * np.square(self.grad_b)

        m_corrected_w = self.m_weights / (1 - self.beta1 ** self.t)
        m_corrected_b = self.m_bias / (1 - self.beta1 ** self.t)
        v_corrected_w = self.v_weights / (1 - self.beta2 ** self.t)
        v_corrected_b = self.v_bias / (1 - self.beta2 ** self.t)


        self.weights = self.weights - self.learning_rate * m_corrected_w / (np.sqrt(v_corrected_w) + self.epsilon)
        self.bias = self.bias - self.learning_rate * m_corrected_b / (np.sqrt(v_corrected_b) + self.epsilon)
    
    def predict_proba(self, X):
        if self.interact:
            X_new = self.interaction(X)
        else:
            X_new = X
        z = np.dot(X_new, self.weights.T) + self.bias
        return  self.sigmoid(z)
    
    def compute_gradients(self, X, y):

        predictions = self.sigmoid(np.dot(X, self.weights.T) + self.bias)
        errors = predictions - y
        self.grad_w = 2 * np.dot(X.T, errors) / len(y)
        self.grad_b = 2 * np.mean(errors)
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
    
    def train(self, X, y, X_val=None, y_val=None):
        if self.interact:
            X_new = self.interaction(X)
            if X_val is not None:
                X_val_new = self.interaction(X_val)
        else:
            X_new = X
            X_val_new = X_val

        if self.weights is None or self.bias is None:
            num_samples, num_features = X_new.shape
            self.weights = np.random.randn(num_features)
            self.bias = 1e-8
        
        for e in range(self.epochs):
            self.compute_gradients(X_new, y)
            self.update_params()

                        # Calculate loss on validation set for early stopping
            if self.early_stopping and X_val_new is not None and y_val is not None:
                val_loss = self.compute_loss(X_val_new, y_val)
                print(val_loss)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print(f"Stopping early at epoch {e+1} as validation loss has not improved for {self.patience} consecutive epochs.")
                        break
    
    def compute_loss(self, X, y):
        predictions = self.predict_proba(X)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss
