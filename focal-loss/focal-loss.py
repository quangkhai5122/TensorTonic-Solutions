import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    loss = - (
        y * ((1 - p) ** gamma) * np.log(p) +
        (1 - y) * (p ** gamma) * np.log(1 - p)
    )

    return np.average(loss)