import math
import numpy as np


def polyn(x):
    return -1 / 4 * ((x - 1) ** 3) + 1 / 2 * x**2 + x


def peri(x):
    return -1 / 2 * np.sin(4 * x) + 1


def comb(x):
    return np.exp(x - 3) * (x - 4) ** 2
