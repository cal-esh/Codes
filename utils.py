import numpy as np

def null(a, rtol=1e-5):
    """Customing null function."""
    if a.size == 0:
        result = np.empty((0,0))
        return result
    else:
        u, s, v = np.linalg.svd(a)
        rank = (s > rtol*s[0]).sum() 
    return v[rank:].T.copy()

def svds(A, k, n):
    """Customing svds function."""
    if A.size == 0:
        U = np.empty((0,n))
        S = np.zeros((n,n))
        Vt = np.empty((0,n)).T
    else:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        U = -U[:, :k]  # Invert the sign and slice to obtain the first k columns
        S = np.diag(S[:k])  # Construct diagonal matrix
        Vt = Vt[:k, :]
    return U, S, Vt