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

    @staticmethod
    def calculateGradientDescent(data, alphaLearningRateDynamic, kLearningRate, stepsNumber,
                                 epsilonLimitation, writeToOutputTxt):
        """Метод градиентного спустка.

        alphaLearningRate = [kLearningRate / i]

        Args:
            data: лист, содержащий входной датасет в виде (area,rooms,price).
            alphaLearningRateDynamic: флаг влючения динамической альфы, зависящий от иттерации: 1 - включен.
            kLearningRate: константа для вычисления alphaLearningRate.
            stepsNumber: максимальное количество иттераций спуска.
            epsilonLimitation: максимальная разница между функционалом ошибки текущей и предыдущей иттераций
             или весами weight_NP[1] и weight_NP[2].
            writeToOutputTxt: флаг записи в таблицы в файл outputGradient.txt: 1 - включен.

        Returns:
            lastIteration: число, последняя иттерация вычислений.
            MSE_hist.tolist(): лист, содержащий значения функционалов ошибок.
            weight_NP.tolist(): лист, содержащий веса w0 для x0, w1 для x1, w2 для x2.
            YNew_NP.tolist(): лист, содержащий гипотезы линейной регрессии.
            weight_hist1(): массив numpy, содержащий веса всех итераций для w1.
            weight_hist2(): массив numpy, содержащий веса всех итераций для w2.
        """
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

        MSE_hist = np.zeros(stepsNumber)
        # weight_hist0 = np.zeros(stepsNumber)
        weight_hist1 = np.zeros(stepsNumber)
        weight_hist2 = np.zeros(stepsNumber)

        i = 0
        while True:
            YNew_NP = X_NP.dot(weight_NP)
            MSE = np.sum((YNew_NP - Y_NP) ** 2) / (2 * N)
            MSE_hist[i] = MSE

            gradient = np.dot(XTranspose_NP, (X_NP.dot(weight_NP) - Y_NP)) / N
            if alphaLearningRateDynamic == 1:
                alphaLearningRate = kLearningRate / (i+1)
            else:
                alphaLearningRate = kLearningRate
            # print("%f %f Iteration %d, MSE(w): %f\n" % (weight_NP[0], weight_NP[1], i, MSE))
            weight_NP = weight_NP - alphaLearningRate * gradient
            # weight_hist0[i] = weight_NP[0]
            weight_hist1[i] = weight_NP[1]
            weight_hist2[i] = weight_NP[2]
            i = i + 1
            if (abs(weight_NP[1] - weight_NP[2]) < epsilonLimitation) or \
                    (abs(MSE_hist[i-1] - MSE_hist[i-2]) < epsilonLimitation) or (stepsNumber == i):
                # print("gradient descent finished")
                lastIteration = i - 1
                if abs(weight_NP[1] - weight_NP[2]) < epsilonLimitation:
                    breakCriterion = "weight"
                if abs(MSE_hist[lastIteration] - MSE_hist[lastIteration-1]) < epsilonLimitation:
                    breakCriterion = "MSE_hist"
                if stepsNumber == i:
                    breakCriterion = "stepsNumber"

                if (writeToOutputTxt == 1):
                    my_file = open('outputGradient.txt', 'a')
                    table = [[breakCriterion, lastIteration, stepsNumber, kLearningRate, alphaLearningRate,
                              epsilonLimitation, np.average(MSE_hist),
                              MSE_hist[lastIteration], MSE_hist[lastIteration - 1], MSE_hist[0],
                              abs(MSE_hist[lastIteration] - MSE_hist[lastIteration - 1]),
                              weight_NP[1], weight_NP[2], abs(weight_NP[1] - weight_NP[2])]]
                    my_file.write(tabulate(table,
                                   # headers=["breakCriterion", "lastIteration",
                                   #          "stepsNumber", "kLearningRate",
                                   #          "current alphaLearningRate",
                                   #          "epsilonLimitation", "errorAvg(MSE)",
                                   #          "MSE[i]", "MSE[i-1]", "MSE[0]", "abs(MSE[i] - MSE[i-1])",
                                   #          "weight_NP[1]", "weight_NP[2]", "abs(weight_NP[1] - weight_NP[2]"],
                                   tablefmt='orgtbl'))
                    my_file.write("\n")
                    my_file.close()
                table = [[breakCriterion, lastIteration, stepsNumber, kLearningRate, alphaLearningRate,
                          epsilonLimitation, np.average(MSE_hist),
                          MSE_hist[lastIteration], MSE_hist[lastIteration-1], MSE_hist[0],
                          abs(MSE_hist[lastIteration] - MSE_hist[lastIteration-1]),
                          weight_NP[1], weight_NP[2], abs(weight_NP[1] - weight_NP[2])]]
                print(tabulate(table,
                               # headers=["breakCriterion", "lastIteration",
                               #          "stepsNumber", "kLearningRate",
                               #          "current alphaLearningRate",
                               #          "epsilonLimitation", "errorAvg(MSE)",
                               #          "MSE[i]", "MSE[i-1]", "MSE[0]", "abs(MSE[i] - MSE[i-1])",
                               #          "weight_NP[1]", "weight_NP[2]", "abs(weight_NP[1] - weight_NP[2]"],
                               tablefmt='orgtbl'))
                return lastIteration, MSE_hist.tolist(), weight_NP.tolist(), YNew_NP.T.tolist(), weight_hist1, \
                       weight_hist2

    @staticmethod
    def calculateInputPrice(areaInputList, roomsInputList, wLast):
        """Метод расчета цены по весам градиентного спуска.

        Args:
            areaInputList: лист, содержащий area составляющую.
            roomsInputList: лист, содержащий rooms составляющую.
            wLast: лист, содержащий веса w0 для x0, w1 для x1, w2 для x2.

        Returns:
            priceNormalizeInputList: лист, содержащий рассчитанные нормализованные цены.
        """
        areaNormalizeInputList, roomsNormalizeInputList = \
            DatasetProcessing.getNormalizeInputDataset(areaInputList, roomsInputList)
        priceNormalizeInputList = []
        for i in range(len(areaInputList)):
            priceNormalizeInputList.append(
                wLast[0][0] + areaNormalizeInputList[i][0] * wLast[1][0] + roomsNormalizeInputList[i][0] * wLast[2][0])
        return priceNormalizeInputList