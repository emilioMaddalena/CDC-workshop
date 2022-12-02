import math
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model._ridge import _solve_cholesky_kernel
from models import KRR


def polyn(x):
    return -1 / 4 * ((x - 1) ** 3) + 1 / 2 * x**2 + x


def peri(x):
    return -1 / 2 * np.sin(4 * x) + 1


def comb(x):
    return np.exp(x - 3) * (x - 4) ** 2


def rkhs(x, lengthscale=1):
    X = np.array([[0], [1], [3], [4], [5], [7]])
    y = np.array([[0], [2], [-1], [1], [-0.5], [4]])
    funct = KRR(1e-8, lengthscale, 0, 6.4)
    funct.fit(X, y)
    print(f"Norm: {funct.get_norm()}")
    return funct.predict(x)
