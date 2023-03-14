import numpy as np

class Oracle():
    def __init__(self, estimator, noise_std):
        """
        Build a noisy oracle as ground truth
        """
        
        self.estimator = estimator
        self.std = noise_std
        
    def predict(self, X):
        y_pred = self.estimator.predict(X)
        noise = np.random.normal(loc = 0, scale = self.std, size = y_pred.shape)
        return y_pred + noise