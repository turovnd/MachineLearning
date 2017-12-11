import numpy as np


class Spearman(object):
    def __init__(self, logs=False):
        self.logs = logs
        self.__coeff = []
        self.__keys = []

    def getCoeff(self):
        return self.__coeff

    def getKeys(self):
        return self.__keys

    def fit(self, X, Y):
        rank_dict_Y, ligaments_Y = get_rank_ligaments(Y)
        for x in X:
            rank_dict_X, ligaments_X = get_rank_ligaments(x)
            n = len(x)
            diff_rank = 0.0
            for i, x in enumerate(x):
                diff_rank += (rank_dict_X[x] - (n + 1) / 2) * (rank_dict_Y[Y[i]] - (n + 1) / 2)
            self.__coeff.append(float(diff_rank) / (n * (n - 1) * (n + 1) - (ligaments_X + ligaments_Y)))

    def process(self):
        spearman_dict = {}
        for i, x in enumerate(self.__coeff):
            spearman_dict[i] = float(np.abs(x))

        self.__keys = sorted(spearman_dict, key=lambda x: spearman_dict[x], reverse=True)

        if self.logs:
            print("\nSpearman")
            print("Distances: ", spearman_dict)
            print("Found keys of distance: ", self.__keys)


def define_rank(Y):
    rank_dict = {}
    count_dict = {}
    y = sorted(Y)
    for i, y_element in enumerate(y):
        rank_dict[y_element] = 0.0
        count_dict[y_element] = 0.0
    for i, y_element in enumerate(y):
        rank_dict[y_element] = i
        count_dict[y_element] += 1.0
    for key in count_dict.keys():
        if count_dict[key] != 1.0:
            rank_dict[key] /= count_dict[key]
    return rank_dict, count_dict


def get_rank_ligaments(Y):
    rank_dict, count_dict = define_rank(Y)
    ligaments = 0.0
    for key in count_dict.keys():
        if count_dict[key] != 1.0:
            ligaments += count_dict[key] * ((count_dict[key] ** 2) - 1.0)
    ligaments *= 1 / 2
    return rank_dict, ligaments
