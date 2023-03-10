import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

def lagrange_experiment(a, b, n, sigma):
    train = np.sort(np.random.uniform(low=a, high=b, size=n))
    epsilon = np.random.normal(loc=0, scale=sigma, size=n)
    poly = lagrange(train + epsilon, np.sin(train))
    test = np.random.uniform(low=a, high=b, size=n)
    train_error = sum((Polynomial(poly.coef[::-1])(train) - np.sin(train)) ** 2) / n
    test_error = sum((Polynomial(poly.coef[::-1])(test) - np.sin(test)) ** 2) / n
    print(n, " & ", train_error, " & ", test_error, " \\\\")

if __name__=="__main__":
    print("Experiment 1\n")
    lagrange_experiment(-3, 3, 10, 0)
    lagrange_experiment(-3, 3, 20, 0)
    lagrange_experiment(-3, 3, 50, 0)
    print("Experiment 2\n")
    lagrange_experiment(-3, 3, 15, .5)
    lagrange_experiment(-3, 3, 15, 1)
    lagrange_experiment(-3, 3, 15, 2)
    lagrange_experiment(-3, 3, 15, 5) 
    pass
