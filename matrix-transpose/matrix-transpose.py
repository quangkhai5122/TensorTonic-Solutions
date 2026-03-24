import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    r, c = A.shape
    res = np.zeros((c,r), dtype=A.dtype)
    for i in range(c):
        for j in range(r):
            res[i][j] = A[j][i]
    return res
    pass
