import numpy as np

class LogisticRegression():
    def __init__(self, alpha=0.001, iters = 300):
        self.alpha = alpha
        self.iters = iters
        self.theta = None

    def sigmoid(self, x):
    # Transforms a linear input into a value between 0 and 1
        return 1 / (1 + np.exp(-x))

    def probability(self, theta, x):
    # Returns the probability after passing through sigmoid
        return self.sigmoid(x.dot(theta))
    
    def fit(self, x, y):
        self.theta = np.zeros((x.shape[1], 1))
        y = y.values.reshape(y.shape[0], 1)
        m = x.shape[0] # no. of comments
        for _ in range(self.iters):
            linear = x.dot(self.theta) # m x 1
            y_pred = self.sigmoid(linear) 
            residual = (y_pred) - y
            dw = (1 / m) * x.T.dot(residual)
            self.theta -= self.alpha * (dw)
        return
    
    def predict(self, x):
        predicted_classes = self.probability(self.theta, x)
        predicted_classes = [1 if i > 0.5 else 0 for i in predicted_classes]
        return predicted_classes