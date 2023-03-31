import numpy as np
import matplotlib.pyplot as plt
from pca import plot_2d_pca, get_reconstruction_error
from dro import dro

data2d = np.loadtxt("homework5/data/data2d.csv", delimiter=',') # shape (50, 2)
data1000d = np.loadtxt("homework5/data/data1000d.csv", delimiter=',') # shape (500, 1000)

def drlv(X, k):
    # initialize params
    b, A, Z = dro(X, k)
    re_X = np.matmul(Z, A.T) + b # reconstructed X
    eta = get_reconstruction_error(X, re_X)
    
    d = X.shape[1]
    X_drlv = X.copy()

    # do 10 iterations of EM
    for i in range(10):
        # E-step
        sigma = np.linalg.inv((np.matmul(A, A.T) + eta * np.identity(d)))
        EZ = np.matmul(np.matmul(A.T, sigma), (X_drlv - b).T).T

        # M-step
        a = np.linalg.inv((eta * np.identity(d)))
        b = np.matmul(EZ.T, (X_drlv - b))
        c = np.linalg.inv((np.matmul(EZ.T, EZ)))
        MA = np.matmul(a, np.matmul(b.T, c))
        az = np.matmul(EZ, A.T)
        Meta = np.matmul((X_drlv - az - b).T, (X_drlv - az - b))

        # Re-assign params
        Z = EZ
        A = MA
        eta = Meta
            
    return b, A, Z, eta

if __name__ == '__main__':
    # experiment

    b2d, A2d, Z2d, eta2d = drlv(data2d, 1)
    X2d = np.matmul(Z2d, A2d.T) + b2d
    print(get_reconstruction_error(data2d, X2d))
    plot_2d_pca(data2d, X2d)
    plt.title("DRLV")
    plt.savefig("homework5/figs/drlv.png")

    b1000d, A1000d, Z1000d, eta1000d = drlv(data1000d, 500)
    X1000d = np.matmul(Z1000d, A1000d.T) + b1000d
    print(get_reconstruction_error(data1000d, X1000d))
    
    pass
