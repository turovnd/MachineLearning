import numpy as np
from numpy import linalg

class Kernel(object):
    def __init__(self):
        pass

    @staticmethod
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    @staticmethod
    def polynomial_kernel(x, y, p=3):
        return (1 + np.dot(x, y)) ** p

    @staticmethod
    def gaussian_kernel(x, y, sigma=5.0):
        return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

    @staticmethod
    def get(kernel):
        if kernel == 'gaussian':
            return Kernel.gaussian_kernel
        elif kernel == 'polynomial':
            return Kernel.polynomial_kernel
        else:
            return Kernel.linear_kernel
