from DatasetProcessing import *
import numpy as np
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
         или весами weight_NP[1] и weight_NP[2].

    Returns:
        lastIteration: число, последняя иттерация вычислений.
        J_hist.tolist(): лист, содержащий значения функционалов ошибок.
        weight_NP.tolist(): лист, содержащий веса w0 для x0, w1 для x1, w2 для x2.
        YNew_NP.tolist(): лист, содержащий гипотезы линейной регрессии.
        weight_hist1(): массив numpy, содержащий веса всех итераций для w1.
        weight_hist2(): массив numpy, содержащий веса всех итераций для w2.
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
        # weight_hist0 = np.zeros(stepsNumber)
        weight_hist1 = np.zeros(stepsNumber)
        weight_hist2 = np.zeros(stepsNumber)

        i = 0
        while True:
            YNew_NP = X_NP.dot(weight_NP)
            J = np.sum((X_NP.dot(weight_NP)) ** 2) / (2 * N)
            J_hist[i] = J

            gradient = np.dot(XTranspose_NP, (X_NP.dot(weight_NP) - Y_NP)) / N

            alphaLearningRate = kLearningRate / (i+1)
            # print("%f %f Iteration %d, J(w): %f\n" % (weight_NP[0], weight_NP[1], i, J))
            weight_NP = weight_NP - alphaLearningRate * gradient
            # weight_hist0[i] = weight_NP[0]
            weight_hist1[i] = weight_NP[1]
            weight_hist2[i] = weight_NP[2]
            i = i + 1
            if (abs(weight_NP[1] - weight_NP[2]) < epsilonLimitation) or \
                    (abs(J_hist[i-1] - J_hist[i-2]) < epsilonLimitation) or (stepsNumber == i):
                # print("-----------------------------------------------------------------------------------------------")
                print("gradient descent finished")
                lastIteration = i - 1
                if abs(weight_NP[1] - weight_NP[2]) < epsilonLimitation:
                    # print("condition (abs(weight_NP[0] - weight_NP[1]) < epsilonLimitation) is done")
                    breakCriterion = "weight"
                if abs(J_hist[lastIteration] - J_hist[lastIteration-1]) < epsilonLimitation:
                    # print("condition (abs(J_hist[i] - J_hist[i-1]) < epsilonLimitation) is done")
                    breakCriterion = "J_hist"
                if stepsNumber == i:
                    # print("condition (stepsNumber == i) is done")
                    breakCriterion = "stepsNumber"

                table = [[breakCriterion, lastIteration, stepsNumber, kLearningRate, alphaLearningRate,
                          epsilonLimitation, np.average(J_hist),
                          J_hist[lastIteration], J_hist[lastIteration-1], J_hist[0], abs(J_hist[lastIteration] - J_hist[lastIteration-1]),
                          weight_NP[1], weight_NP[2], abs(weight_NP[1] - weight_NP[2])]]
                print(tabulate(table,
                               headers=["breakCriterion", "lastIteration",
                                        "stepsNumber", "kLearningRate",
                                        "current alphaLearningRate",
                                        "epsilonLimitation", "errorAvg(MSE)",
                                        "MSE[i]", "MSE[i-1]", "MSE[0]", "abs(MSE[i] - MSE[i-1])",
                                        "weight_NP[1]", "weight_NP[2]", "abs(weight_NP[1] - weight_NP[2]"],
                               tablefmt='orgtbl'))

                # print("-----------------------------------------------------------------------------------------------")
                return lastIteration, J_hist.tolist(), weight_NP.tolist(), YNew_NP.T.tolist(), weight_hist1, \
                       weight_hist2

    """Метод расчета цены по весам градиентного спуска.

    Args:
        areaInputList: лист, содержащий area составляющую.
        roomsInputList: лист, содержащий rooms составляющую.
        wLast: лист, содержащий веса w0 для x0, w1 для x1, w2 для x2.
        
    Returns:
        priceNormalizeInputList: лист, содержащий рассчитанные нормализованные цены.
    """
    @staticmethod
    def calculateInputPrice(areaInputList, roomsInputList, wLast):
        areaNormalizeInputList, roomsNormalizeInputList = \
            DatasetProcessing.getNormalizeInputDataset(areaInputList, roomsInputList)
        priceNormalizeInputList = []
        for i in range(len(areaInputList)):
            priceNormalizeInputList.append(
                wLast[0][0] + areaNormalizeInputList[i][0] * wLast[1][0] + roomsNormalizeInputList[i][0] * wLast[2][0])
        return priceNormalizeInputList