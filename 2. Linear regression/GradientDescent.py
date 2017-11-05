from DatasetProcessing import *
import numpy as np
import pprint as pp
"""
"""

class GradientDescent(object):
    """Initialization variables"""
    def __init__(self, alphaLearningRate, stepsNumber, epsilonLimitation):
        self.alphaLearningRate = alphaLearningRate
        self.stepsNumber = stepsNumber
        self.epsilonLimitation = epsilonLimitation

    @staticmethod
    def y_hat(X, weight):
        """
        Linear regression hypothesis: y_hat = X.w
        """

        return X.dot(weight)

    @staticmethod
    def calculateNewPrice(data, alphaLearningRate, stepsNumber):
        normalizeData = DatasetProcessing.getNormalizeDataset(data)
        area, rooms, Y = DatasetProcessing.getSeparetedData(normalizeData)
        """
        weight = [[0]*100, [0]*100]
        priceNew = []

        J_history = []
        for k in range(0, stepsNumber):
            for j in range(len(area)):
                temp = area[j]*weight[0][j] + rooms[j]*weight[1][j]
                priceNew.append(temp)
            # return priceNew
            for j in range(len(area)):
                J = sum((priceNew[j] - price[j])**2) / (2 * len(area))
                J_history[j].append(J)
                gradient =
        """
        """
        ### example 1
        a = np.array((1,2))
        b = np.array((3,4))
        print("1 ", np.dot(a,b))  # 11
        print("2 ", a*b)  # [3 8]
        ### example 2
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([[10], [20], [30]])
        xt = x.T
        w = np.array([[0], [0], [0]])
        J = np.sum((x.dot(w) - y) ** 2) / (2 * 13)
        gradient = np.dot(xt,x.dot(w) - y) / 13
        w = w - alphaLearningRate*gradient
        print(w)
        """


        X_NP = np.vstack((area, rooms))  # двумерный массив [,.....,],[,.....,]
        XTranspose_NP = np.transpose(X_NP)

        YBad_NP = np.asarray(Y)
        YBad_NP = YBad_NP.reshape((1, -1)) # transpose feature
        Y_NP = np.transpose(YBad_NP)

        m, n = np.shape(XTranspose_NP) # !
        N = Y_NP.shape[0]

        weight_NP = np.array([np.ones(n)]).T
        # ################################
        # print("X:")
        # print(XTranspose_NP)
        #
        # print("\nY:")
        # print(Y_NP)
        #
        # print("\nw:")
        # print(weight_NP)
        #print(weight_NP)
        #################################
        J_hist = np.zeros(stepsNumber)
        for i in range(0, stepsNumber):
            J = np.sum((XTranspose_NP.dot(weight_NP)) ** 2) / (2 * N)
            # J = np.sum((X_NP.dot(weight_NP) - Y_NP) ** 2) / (2 * N)
            J_hist[i] = J
            print("Iteration %d, J(w): %f\n" % (i, J))
            gradient = np.dot(X_NP, (XTranspose_NP.dot(weight_NP) - Y_NP)) / N

            weight_NP = weight_NP - alphaLearningRate * gradient
        return (J_hist, weight_NP)