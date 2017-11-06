from DatasetProcessing import *
import numpy as np
import pprint as pp
import matplotlib.pyplot as plt
from tabulate import tabulate
"""
"""

class GradientDescent(object):
    """Initialization variables"""
    def __init__(self, kLearningRate, stepsNumber, epsilonLimitation):
        self.kLearningRate = kLearningRate
        self.stepsNumber = stepsNumber
        self.epsilonLimitation = epsilonLimitation

    """Метод градиентного спустка.
    
    alphaLearningRate = [kLearningRate / i]
    
    Args:
        data: лист, содержащий входной датасет в виде (area,rooms,price).
        kLearningRate: константа для вычисления alphaLearningRate.
        stepsNumber: максимальное количество иттераций спуска.
        epsilonLimitation: максимальная разница между функционалом ошибки текущей и предыдущей иттераций.

    Returns:
        stepsNumber:
        J_hist: лист, содержащий значения функционалов ошибок.
        weight_NP: лист, содержащий две последних ошибки.
        YNew_NP: лист, содержащий гипотезы линейной регрессии.
    """
    @staticmethod
    def calculateGradientDescent(data, kLearningRate, stepsNumber, epsilonLimitation):
        normalizeData = DatasetProcessing.getNormalizeDataset(data)
        area, rooms, Y = DatasetProcessing.getSeparetedData(normalizeData)

        X_NP = np.vstack((area, rooms))  # двумерный массив [,.....,],[,.....,]
        XTranspose_NP = np.transpose(X_NP)

        YBad_NP = np.asarray(Y)
        YBad_NP = YBad_NP.reshape((1, -1))  # transpose feature
        Y_NP = np.transpose(YBad_NP)

        m, n = np.shape(XTranspose_NP)
        N = Y_NP.shape[0]

        weight_NP = np.array([np.ones(n)]).T

        J_hist = np.zeros(stepsNumber)
        i = 0
        while True:
            YNew_NP = XTranspose_NP.dot(weight_NP)
            J = np.sum((XTranspose_NP.dot(weight_NP)) ** 2) / (2 * N)
            J_hist[i] = J
            print("Iteration %d, J(w): %f\n" % (i, J))
            gradient = np.dot(X_NP, (XTranspose_NP.dot(weight_NP) - Y_NP)) / N

            i = i + 1
            alphaLearningRate = kLearningRate / i
            weight_NP = weight_NP - alphaLearningRate * gradient
            if (abs(J_hist[i-1] - J_hist[i-2]) < epsilonLimitation) or (stepsNumber == i):
                print("-----------------------------------------------------------------------------------------------")
                print("gradient descent finished")
                if abs(J_hist[i-1] - J_hist[i-2]) < epsilonLimitation:
                    print("condition (abs(weight_NP[0] - weight_NP[1]) < epsilonLimitation) is done")
                if stepsNumber == i:
                    print("condition (stepsNumber == i) is done")
                table = [[i, stepsNumber, kLearningRate, alphaLearningRate, epsilonLimitation,
                          abs(J_hist[i-1] - J_hist[i-2]), J_hist[i-1], J_hist[i-2]]]
                print(tabulate(table,
                               headers=["iteration",
                                        "stepsNumber", "kLearningRate",
                                        "current alphaLearningRate",
                                        "epsilonLimitation", "abs(J_hist[i-1] - J_hist[i-2])", "J_hist[i-1]",
                                        "J_hist[i-2]"],
                               tablefmt='orgtbl'))
                print("-----------------------------------------------------------------------------------------------")
                return J_hist.tolist(), weight_NP.tolist(), YNew_NP.T.tolist()