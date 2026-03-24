import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    r, c = A.shape
    m = max(r,c)
    res = np.zeros((m,m), dtype=A.dtype)
    res[:r, :c] = A
    for i in range(m):
        for j in range(m):
            if i < j:
                tmp = res[i][j]
                res[i][j] = res[j][i]
                res[j][i] = tmp
    return res[:c, :r]
    pass
