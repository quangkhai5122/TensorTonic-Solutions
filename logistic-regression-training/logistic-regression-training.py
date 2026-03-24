import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def gradient(x, y, w, b):
    p = _sigmoid(np.dot(w, x) + b)
    dw = (p - y) * x
    db = (p - y)
    return dw, db

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for i in range(steps):
        for j in range(len(X)):
            dw, db = gradient(X[j], y[j], w, b)
            w = w - lr*dw
            b = b - lr*db
    return (w, b)
    pass