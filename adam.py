import numpy as np
import math
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(17)

class LogisticRegressionAdam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epochs=100, epsilon=1e-8, early_stopping=False, patience=10):
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
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_weights = None
        self.best_bias = None
        self.best_accuracy = 0
        self.epochs_no_improve = 0
    
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
        z = np.dot(X, self.weights.T) + self.bias
        return  self.sigmoid(z)
    
    def compute_gradients(self, X, y):
        predictions = self.predict_proba(X)
        errors = predictions - y
        self.grad_w = 2 * np.dot(X.T, errors) / len(y)
        self.grad_b = 2 * np.mean(errors)

    def compute_loss(self, X, y):
        predictions = self.predict_proba(X)
        return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        if X_valid is None or y_valid is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=17)
        if self.weights is None or self.bias is None:
            num_samples, num_features = X_train.shape
            self.weights = np.random.randn(num_features)
            self.bias = 1e-8

        self.best_loss = np.inf
        self.train_losses = []
        self.valid_losses = []
        self.train_balanced_accuracies = []
        self.valid_balanced_accuracies = []


        for e in range(self.epochs):
            self.compute_gradients(X_train, y_train)
            self.update_params()

            train_loss = self.compute_loss(X_train, y_train)
            valid_loss = self.compute_loss(X_valid, y_valid)

            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            y_pred_train = self.predict(X_train)
            y_pred_valid = self.predict(X_valid)
            valid_balanced_accuracy = balanced_accuracy_score(y_valid, y_pred_valid)
            train_balanced_accuracy = balanced_accuracy_score(y_train, y_pred_train)

            self.train_balanced_accuracies.append(train_balanced_accuracy)
            self.valid_balanced_accuracies.append(valid_balanced_accuracy)

            if self.early_stopping:
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.best_weights = self.weights.copy()
                    self.best_bias = self.bias
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve == self.patience:
                        print(f'Stopping early at epoch {e}')
                        self.weights = self.best_weights
                        self.bias = self.best_bias
                        print(f'Best loss: {self.best_loss}')
                        break
