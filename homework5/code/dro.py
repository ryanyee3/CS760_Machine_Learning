import numpy as np
import matplotlib.pyplot as plt

data2d = np.loadtxt("homework5/data/data2d.csv", delimiter=',') # shape (50, 2)
data1000d = np.loadtxt("homework5/data/data1000d.csv", delimiter=',') # shape (500, 1000)

def dro(X, k):
    d = X.shape[1]
    X_dro = X.copy()
    b = np.mean(X_dro, axis=0)
    Q = X_dro - b
    u, s, vt = np.linalg.svd(Q)
    A = vt[range(k)].reshape((d, k))
    Z = np.matmul(Q, A)
    return b, A, Z

def plot_2d_pca(X, re_X):
    plt.scatter(X[:, 0], X[:, 1], c="blue", marker='x')
    plt.scatter(re_X[:, 0], re_X[:, 1], c="red", marker='+')

def get_reconstruction_error(X, re_X):
    return round((np.linalg.norm(X - re_X)**2) / len(X), 6)

# experiment
b2d, A2d, Z2d = dro(data2d, 1)
X2d = np.matmul(Z2d, A2d.T) + b2d
print(get_reconstruction_error(data2d, X2d))
plot_2d_pca(data2d, X2d)
plt.title("DRO")
plt.savefig("homework5/figs/dro.png")

b1000d, A1000d, Z1000d = dro(data1000d, 500)
X1000d = np.matmul(Z1000d, A1000d.T) + b1000d
print(get_reconstruction_error(data1000d, X1000d))

if __name__ == '__main__':
    pass
