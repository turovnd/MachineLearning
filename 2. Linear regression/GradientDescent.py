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
        epsilonLimitation: максимальная разница между функционалом ошибки текущей и предыдущей иттераций
         или весами weight_NP[0] и weight_NPх[1].

    Returns:
        J_hist: лист, содержащий значения функционалов ошибок.
        weight_NP: лист, содержащий две последних ошибки.
        YNew_NP: лист, содержащий гипотезы линейной регрессии.
        weight_hist0: лист, содержащий все веса для первого параметра.
        weight_hist1: лист, содержащий все веса для второго параметра.
    """
    @staticmethod
    def calculateGradientDescent(data, kLearningRate, stepsNumber, epsilonLimitation):
        normalizeData = DatasetProcessing.getNormalizeDataset(data)
        area, rooms, Y = DatasetProcessing.getSeparetedData(normalizeData)

        x0 = np.ones(len(area))

        XTranspose_NP = np.vstack((x0, area, rooms))  # двумерный массив [,.....,],[,.....,]
        X_NP = np.transpose(XTranspose_NP)

        YBad_NP = np.asarray(Y)
        YBad_NP = YBad_NP.reshape((1, -1))  # transpose feature
        Y_NP = np.transpose(YBad_NP)

        m, n = np.shape(X_NP)
        N = Y_NP.shape[0]

        weight_NP = np.array([np.ones(n)]).T

        J_hist = np.zeros(stepsNumber)
        weight_hist0 = np.zeros(stepsNumber)
        weight_hist1 = np.zeros(stepsNumber)
        i = 0
        while True:
            YNew_NP = X_NP.dot(weight_NP)
            J = np.sum((X_NP.dot(weight_NP)) ** 2) / (2 * N)
            J_hist[i] = J

            gradient = np.dot(XTranspose_NP, (X_NP.dot(weight_NP) - Y_NP)) / N


            alphaLearningRate = kLearningRate / (i+1)
            # print("%f %f Iteration %d, J(w): %f\n" % (weight_NP[0], weight_NP[1], i, J))
            weight_NP = weight_NP - alphaLearningRate * gradient
            weight_hist0[i] = weight_NP[0]
            weight_hist1[i] = weight_NP[1]
            i = i + 1
            if (abs(weight_NP[0] - weight_NP[1]) < epsilonLimitation) or \
                    (abs(J_hist[i-1] - J_hist[i-2]) < epsilonLimitation) or (stepsNumber == i):
                print("-----------------------------------------------------------------------------------------------")
                print("gradient descent finished")
                if abs(weight_NP[0] - weight_NP[1]) < epsilonLimitation:
                    print("condition (abs(weight_NP[0] - weight_NP[1]) < epsilonLimitation) is done")
                if abs(J_hist[i-1] - J_hist[i-2]) < epsilonLimitation:
                    print("condition (abs(J_hist[i-1] - J_hist[i-2]) < epsilonLimitation) is done")
                if stepsNumber == i:
                    print("condition (stepsNumber == i) is done")
                table = [[i, stepsNumber, kLearningRate, alphaLearningRate, epsilonLimitation,
                          J_hist[i-1], J_hist[i-2], abs(J_hist[i-1] - J_hist[i-2]),
                          weight_NP[0], weight_NP[1], abs(weight_NP[0] - weight_NP[1])]]
                print(tabulate(table,
                               headers=["iteration",
                                        "stepsNumber", "kLearningRate",
                                        "current alphaLearningRate",
                                        "epsilonLimitation",
                                        "J_hist[i-1]", "J_hist[i-2]", "abs(J_hist[i-1] - J_hist[i-2])",
                                        "weight_NP[0]", "weight_NP[1]", "abs(weight_NP[0] - weight_NP[1]"],
                               tablefmt='orgtbl'))
                print("-----------------------------------------------------------------------------------------------")
                return J_hist.tolist(), weight_NP.tolist(), YNew_NP.T.tolist(), weight_hist0.tolist(), \
                       weight_hist1.tolist()