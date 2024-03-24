import numpy as np

class LogisticRegressionAdam:
    def __init__(self, learning_rate=0.02, beta1=0.9, beta2=0.999, epochs = 100, epsilon=1e-8):
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
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
    
    def train(self, X, y):

        if self.weights is None or self.bias is None:
            num_samples, num_features = X.shape
            self.weights = np.random.randn(num_features)
            self.bias = 1e-8

        for e in range(self.epochs):
            self.compute_gradients(X, y)
            self.update_params()
