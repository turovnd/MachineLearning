import numpy as np
import math


class IG(object):
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
            Hc = get_entropy_Y(Y)
            Sum = get_entropy_X(x)
            M = get_entropy_Y_to_X(x, Y)
            H_C_E = (-1) * Sum * (-1) * M
            self.__coeff.append(Hc - H_C_E)

    def process(self):
        IG_dict = {}

        for i, x in enumerate(self.__coeff):
            IG_dict[i] = x

        self.__keys = sorted(IG_dict, key=lambda x: IG_dict[x], reverse=True)

        if self.logs:
            print("\nIG")
            print("Distances: ", IG_dict)
            print("Found keys of distance: ", self.__keys)


def get_entropy_Y(Y):
    positive_class_count = 0
    negative_class_count = 0
    for y_element in Y:
        if y_element > 0:
            positive_class_count += 1
        if y_element < 0:
            negative_class_count += 1
    positive_probability = float(positive_class_count) / len(Y)
    negative_probability = float(negative_class_count) / len(Y)
    entropy = positive_probability * math.log(positive_probability, 2) + \
              negative_probability * math.log(negative_probability, 2)
    return entropy


def get_entropy_X(X):
    x_count_dict = {}
    n = len(X)
    entropy = 0.0
    for x_element in X:
        x_count_dict[x_element] = 0.0
    for x_element in X:
        x_count_dict[x_element] += 1.0
    for key in x_count_dict.keys():
        probability = x_count_dict[key] / n
        entropy += probability * math.log(probability, 2)
    return entropy


def get_entropy_Y_to_X(X, Y):
    dict_positive, dict_negative = get_matrix_Y_X(X, Y)
    M = 0
    for i, y_element in enumerate(Y):
        if y_element > 0:
            probability = dict_positive[X[i]] / (dict_positive[X[i]] + dict_negative[X[i]])
            M += probability * math.log(probability, 2)
        if y_element < 0:
            probability = dict_negative[X[i]] / (dict_positive[X[i]] + dict_negative[X[i]])
            M += probability * math.log(probability, 2)
    return M


def get_matrix_Y_X(X, Y):
    dict_positive = {}
    dict_negative = {}
    for x_element in X:
        dict_positive[x_element] = 0.0
        dict_negative[x_element] = 0.0
    for i, x_element in enumerate(X):
        if Y[i] < 0:
            dict_negative[x_element] += 1.0
        else:
            dict_positive[x_element] += 1.0
    return dict_positive, dict_negative
