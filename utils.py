import numpy as np

def Laplacian_inv(M, p, n):
    '''
    Computes the inverse operation of L
    M : shape p*p
    w : shape p*(p - 1)/2
    '''
    ind = 0
    w = np.zeros(n)
    for i in range(0, p):
        for j in range(i+1, p):
            w[ind] = -M[i][j]
            ind += 1
    return w

def Laplacian(w, p):
    """
    Compute laplacian from a vector

    w :  shape p*(p-1)/2
    LW : shape p*p
    """
    Lw = np.zeros((p, p))
    ind = 0
    for i in range(p-1):
        for j in range(i+1, p):
            Lw[i, j] = -w[ind]
            ind += 1
    Lw = Lw + Lw.T
    np.fill_diagonal(Lw, -(Lw.sum(axis = 0) - np.trace(Lw)))
    return Lw

def Laplacian_dual(M, p, n):
    """
    Compute the dual of L operator

    M :  shape p*p
    (L*)M : shape p*(p-1)/2
    """
    L_Y = np.zeros(n)
    for j in range(p):
        for i in range(j+1, p):
            k = np.int(i - j + j*(2*p - j - 1)/2) - 1
            L_Y[k] = M[i, i] - M[i, j] - M[j, i] + M[j, j]
    return L_Y
