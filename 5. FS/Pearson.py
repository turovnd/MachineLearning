import numpy as np
import math


class Pearson(object):

    def __init__(self, X, Y, logs=False):
        self.logs = logs
        self.__X = X
        self.__Y = Y
        self.__coeff = []
        self.__keys = []

    def getCoeff(self):
        return self.__coeff

    def getKeys(self):
        return self.__keys

    def fit(self):
        Y = self.__Y
        for X in self.__X:
            x_mean = X.mean()
            y_mean = Y.mean()
            numerator = 0
            x_2 = 0
            y_2 = 0
            for i in range(len(X)):
                x_element = X[i] - x_mean
                y_element = Y[i] - y_mean
                numerator += x_element * y_element
                x_2 += x_element ** 2
                y_2 += y_element ** 2
            if x_2 == 0 or y_2 == 0:
                self.__coeff.append(1)
            else:
                self.__coeff.append(numerator / math.sqrt(x_2 * y_2))

    def process(self):
        pearson_dict = {}

        for i, x in enumerate(self.__coeff):
            pearson_dict[i] = np.abs(x)

        self.__keys = sorted(pearson_dict, key=lambda x: pearson_dict[x], reverse=True)

        if self.logs:
            print("\nPearson")
            print("Distances: ", pearson_dict)
            print("Found keys of distance: ", self.__keys)
