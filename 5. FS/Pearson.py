import numpy as np
import math


class Pearson(object):

    def __init__(self, logs=False):
        self.logs = logs
        self.__coeff = []
        self.__keys = []

    def getCoeff(self):
        return self.__coeff

    def getKeys(self):
        return self.__keys

    def fit(self, X, Y):
        for x in X:
            sum_up = np.sum((x - x.mean()) * (Y - Y.mean()))
            sum_down = np.sum((x - x.mean()) ** 2) * np.sum((Y - Y.mean()) ** 2)
            if sum_down == 0:
                self.__coeff.append(0)
            else:
                self.__coeff.append(sum_up / np.sqrt(sum_down))

    def process(self):
        pearson_dict = {}

        for i, x in enumerate(self.__coeff):
            pearson_dict[i] = np.abs(x)

        self.__keys = sorted(pearson_dict, key=lambda x: pearson_dict[x], reverse=True)

        if self.logs:
            print("\nPearson")
            print("Distances: ", pearson_dict)
            print("Found keys of distance: ", self.__keys)
