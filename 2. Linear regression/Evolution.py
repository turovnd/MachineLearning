from DatasetProcessing import *
from Visualization import *
from GradientDescent import *
import numpy as np
from tabulate import tabulate
"""
"""

class Evolution(object):
    """Initialization variables"""
    def __init__(self):
        pass

    @staticmethod
    def startEvolution(numberOfIteration):
        weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4, weight_NP5, weight_NP6 = \
            Evolution.weightStartIntialization()

        MSE_histTop0 = np.zeros(numberOfIteration)
        MSE_histTop1 = np.zeros(numberOfIteration)
        MSE_histTop2 = np.zeros(numberOfIteration)

        MSE_hist0 = np.zeros(numberOfIteration)
        MSE_hist1 = np.zeros(numberOfIteration)
        MSE_hist2 = np.zeros(numberOfIteration)
        MSE_hist3 = np.zeros(numberOfIteration)
        MSE_hist4 = np.zeros(numberOfIteration)
        MSE_hist5 = np.zeros(numberOfIteration)
        MSE_hist6 = np.zeros(numberOfIteration)
        for i in range(numberOfIteration):
            # MSE0, MSE1, MSE2, MSE3, MSE4, MSE5, MSE6 = Evolution.testEvo(i)
            MSE_top0, MSE_top1, MSE_top2, MSE0, MSE1, MSE2, MSE3, MSE4, MSE5, MSE6 = Evolution.testEvo(i)
            MSE_histTop0[i] = MSE_top0
            MSE_histTop1[i] = MSE_top1
            MSE_histTop2[i] = MSE_top2

            MSE_hist0[i] = MSE0
            MSE_hist1[i] = MSE1
            MSE_hist2[i] = MSE2
            MSE_hist3[i] = MSE3
            MSE_hist4[i] = MSE4
            MSE_hist5[i] = MSE5
            MSE_hist6[i] = MSE6
        # Visualization.build2DMSEEvolution(MSE_hist0, MSE_hist1, MSE_hist2, MSE_hist3, MSE_hist4, MSE_hist5, MSE_hist6,
        #                                   numberOfIteration)
        print(type(MSE_hist0))
        print(MSE_hist0)
        Visualization.build2DTopMSEEvolution(MSE_histTop0, MSE_histTop1, MSE_histTop2, numberOfIteration)
        Visualization.build2DIndividualMSEEvolution(MSE_hist0, MSE_hist1, MSE_hist2, MSE_hist3, MSE_hist4, MSE_hist5,
                                                    MSE_hist6, numberOfIteration)

    @staticmethod
    def testEvo(iteration):
        # print(wLast) = print(np.random.rand(3, 1).tolist())
        weight_NP0 = np.random.randn(3, 1)
        weight_NP1 = np.random.randn(3, 1)
        weight_NP2 = np.random.randn(3, 1)
        weight_NP3 = np.random.randn(3, 1)
        weight_NP4 = np.random.randn(3, 1)
        weight_NP5 = np.random.randn(3, 1)
        weight_NP6 = np.random.randn(3, 1)
        data = DatasetProcessing.getDataset('dataset.txt')
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

        MSE0 = np.sum((X_NP.dot(weight_NP0)) ** 2) / (2 * N)
        MSE1 = np.sum((X_NP.dot(weight_NP1)) ** 2) / (2 * N)
        MSE2 = np.sum((X_NP.dot(weight_NP2)) ** 2) / (2 * N)
        MSE3 = np.sum((X_NP.dot(weight_NP3)) ** 2) / (2 * N)
        MSE4 = np.sum((X_NP.dot(weight_NP4)) ** 2) / (2 * N)
        MSE5 = np.sum((X_NP.dot(weight_NP5)) ** 2) / (2 * N)
        MSE6 = np.sum((X_NP.dot(weight_NP6)) ** 2) / (2 * N)

        MSE_total = []
        MSE_total.append(MSE0)
        MSE_total.append(MSE1)
        MSE_total.append(MSE2)
        MSE_total.append(MSE3)
        MSE_total.append(MSE4)
        MSE_total.append(MSE5)
        MSE_total.append(MSE6)
        MSE_return0 = MSE0
        MSE_return1 = MSE1
        MSE_return2 = MSE2
        MSE_return3 = MSE3
        MSE_return4 = MSE4
        MSE_return5 = MSE5
        MSE_return6 = MSE6

        MSE_top = []
        MSE_place = []
        # отбор: поиск лучшего MSE
        for k in range(3):
            for j in range(len(MSE_total)):
                if MSE_total[j] == min(MSE_total):
                    # print(j, " - ", MSE_total[j])
                    MSE_place.append(j)
                    MSE_top.append(MSE_total[j])
                    MSE_total[j] = 10000.0
                    break
        # print(MSE_top)
        # print(MSE_place)


        # отбор: вывод лучших весов по лучшим MSE
        weight_top = []
        for numberMSE_place in range(3):
            # for numberWeight in range(7):
            if MSE_place[numberMSE_place] == 0:
                # print("weight_NP0", weight_NP0)
                weight_top.append(weight_NP0.tolist())
            if MSE_place[numberMSE_place] == 1:
                # print("weight_NP1", weight_NP1)
                weight_top.append(weight_NP1.tolist())
            if MSE_place[numberMSE_place] == 2:
                # print("weight_NP2", weight_NP2)
                weight_top.append(weight_NP2.tolist())
            if MSE_place[numberMSE_place] == 3:
                # print("weight_NP3", weight_NP3)
                weight_top.append(weight_NP3.tolist())
            if MSE_place[numberMSE_place] == 4:
                # print("weight_NP4", weight_NP4)
                weight_top.append(weight_NP4.tolist())
            if MSE_place[numberMSE_place] == 5:
                # print("weight_NP5", weight_NP5)
                weight_top.append(weight_NP5.tolist())
            if MSE_place[numberMSE_place] == 6:
                # print("weight_NP6", weight_NP6)
                weight_top.append(weight_NP6.tolist())
        # print(weight_top, "\n")
        # print(weight_top[0])
        # print("------------------------------")
        table = [[iteration, MSE_top[0], MSE_top[1], MSE_top[2], MSE_total[0], MSE_total[1], MSE_total[2], MSE_total[3],
                  MSE_total[4], MSE_total[5], MSE_total[6]]]
        print(tabulate(table,
                       # headers=["MSE_top[0]", "MSE_top[1]",
                       #          "MSE_top[2]", "kLearningRate",
                       #          "current alphaLearningRate",
                       #          "epsilonLimitation", "errorAvg(MSE)",
                       #          "MSE[i]", "MSE[i-1]", "MSE[0]", "abs(MSE[i] - MSE[i-1])",
                       #          "weight_NP[1]", "weight_NP[2]", "abs(weight_NP[1] - weight_NP[2]"],
                       tablefmt='orgtbl'))
        weight_NP0 = np.asarray(weight_top[0])
        # print("weight_NP0", weight_NP0)
        # w1 [0]
        weight_NP1_1 = np.asarray(weight_top[0])
        weight_NP1_2 = np.asarray(weight_top[1])
        weight_NP1_1List = weight_NP1_1.tolist()
        weight_NP1_2List = weight_NP1_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP1_1List[0][0]))
        string1 = str(float(weight_NP1_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w1 [1]
        string0 = str(float(weight_NP1_1List[1][0]))
        string1 = str(float(weight_NP1_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w1 [2]
        string0 = str(float(weight_NP1_1List[2][0]))
        string1 = str(float(weight_NP1_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        weight_NP1 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP1", weight_NP1)

        # w2 [0]
        weight_NP2_1 = np.asarray(weight_top[1])
        weight_NP2_2 = np.asarray(weight_top[0])
        weight_NP2_1List = weight_NP2_1.tolist()
        weight_NP2_2List = weight_NP2_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP2_1List[0][0]))
        string1 = str(float(weight_NP2_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w2 [1]
        string0 = str(float(weight_NP2_1List[1][0]))
        string1 = str(float(weight_NP2_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w2 [2]
        string0 = str(float(weight_NP2_1List[2][0]))
        string1 = str(float(weight_NP2_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        weight_NP2 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP2", weight_NP2)

        # w3 [0]
        weight_NP3_1 = np.asarray(weight_top[0])
        weight_NP3_2 = np.asarray(weight_top[2])
        weight_NP3_1List = weight_NP3_1.tolist()
        weight_NP3_2List = weight_NP3_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP3_1List[0][0]))
        string1 = str(float(weight_NP3_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w3 [1]
        string0 = str(float(weight_NP3_1List[1][0]))
        string1 = str(float(weight_NP3_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w3 [2]
        string0 = str(float(weight_NP3_1List[2][0]))
        string1 = str(float(weight_NP3_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        weight_NP3 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP3", weight_NP3)

        # w4 [0]
        weight_NP4_1 = np.asarray(weight_top[2])
        weight_NP4_2 = np.asarray(weight_top[0])
        weight_NP4_1List = weight_NP4_1.tolist()
        weight_NP4_2List = weight_NP4_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP4_1List[0][0]))
        string1 = str(float(weight_NP4_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w4 [1]
        string0 = str(float(weight_NP4_1List[1][0]))
        string1 = str(float(weight_NP4_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w4 [2]
        string0 = str(float(weight_NP4_1List[2][0]))
        string1 = str(float(weight_NP4_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        weight_NP4 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP4", weight_NP4)

        # w5 [0]
        weight_NP5_1 = np.asarray(weight_top[2])
        weight_NP5_2 = np.asarray(weight_top[1])
        weight_NP5_1List = weight_NP5_1.tolist()
        weight_NP5_2List = weight_NP5_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP5_1List[0][0]))
        string1 = str(float(weight_NP5_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w5 [1]
        string0 = str(float(weight_NP5_1List[1][0]))
        string1 = str(float(weight_NP5_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w5 [2]
        string0 = str(float(weight_NP5_1List[2][0]))
        string1 = str(float(weight_NP5_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        weight_NP5 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP5", weight_NP5)

        # w6 [0]
        weight_NP6_1 = np.asarray(weight_top[1])
        weight_NP6_2 = np.asarray(weight_top[2])
        weight_NP6_1List = weight_NP6_1.tolist()
        weight_NP6_2List = weight_NP6_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP6_1List[0][0]))
        string1 = str(float(weight_NP6_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w6 [1]
        string0 = str(float(weight_NP6_1List[1][0]))
        string1 = str(float(weight_NP6_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        # w6 [2]
        string0 = str(float(weight_NP6_1List[2][0]))
        string1 = str(float(weight_NP6_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12:]
        weight_NP6 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP6", weight_NP6)
        # return MSE_return[0], MSE_return[1], MSE_return[2], MSE_return[3], MSE_return[4], MSE_return[5], MSE_return[6]
        return MSE_top[0], MSE_top[1], MSE_top[2], MSE_return0, MSE_return1, MSE_return2, MSE_return3, \
               MSE_return4, MSE_return5, MSE_return6

    @staticmethod
    def weightStartIntialization():
        # print(wLast) = print(np.random.rand(3, 1).tolist())
        weight_NP0 = np.random.randn(3, 1)
        weight_NP1 = np.random.randn(3, 1)
        weight_NP2 = np.random.randn(3, 1)
        weight_NP3 = np.random.randn(3, 1)
        weight_NP4 = np.random.randn(3, 1)
        weight_NP5 = np.random.randn(3, 1)
        weight_NP6 = np.random.randn(3, 1)
        return weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4, weight_NP5, weight_NP6

    @staticmethod
    def calculateMSE(weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4, weight_NP5, weight_NP6):
        data = DatasetProcessing.getDataset('dataset.txt')
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
        # stepsNumber = 100000  ##
        # MSE_hist0 = np.zeros(stepsNumber)
        # MSE_hist1 = np.zeros(stepsNumber)
        # MSE_hist2 = np.zeros(stepsNumber)
        # MSE_hist3 = np.zeros(stepsNumber)
        # MSE_hist4 = np.zeros(stepsNumber)
        # MSE_hist5 = np.zeros(stepsNumber)
        # MSE_hist6 = np.zeros(stepsNumber)
        # i = 0
        # while True:
        YNew_NP0 = X_NP.dot(weight_NP0)
        YNew_NP1 = X_NP.dot(weight_NP1)
        YNew_NP2 = X_NP.dot(weight_NP2)
        YNew_NP3 = X_NP.dot(weight_NP3)
        YNew_NP4 = X_NP.dot(weight_NP4)
        YNew_NP5 = X_NP.dot(weight_NP5)
        YNew_NP6 = X_NP.dot(weight_NP6)

        MSE0 = np.sum((X_NP.dot(weight_NP0)) ** 2) / (2 * N)
        MSE1 = np.sum((X_NP.dot(weight_NP1)) ** 2) / (2 * N)
        MSE2 = np.sum((X_NP.dot(weight_NP2)) ** 2) / (2 * N)
        MSE3 = np.sum((X_NP.dot(weight_NP3)) ** 2) / (2 * N)
        MSE4 = np.sum((X_NP.dot(weight_NP4)) ** 2) / (2 * N)
        MSE5 = np.sum((X_NP.dot(weight_NP5)) ** 2) / (2 * N)
        MSE6 = np.sum((X_NP.dot(weight_NP6)) ** 2) / (2 * N)

        # MSE_hist0[i] = MSE0
        # MSE_hist1[i] = MSE1
        # MSE_hist2[i] = MSE2
        # MSE_hist3[i] = MSE3
        # MSE_hist4[i] = MSE4
        # MSE_hist5[i] = MSE5
        # MSE_hist6[i] = MSE6
        # i = i + 1

        MSE_total = []
        MSE_total.append(MSE0)
        MSE_total.append(MSE1)
        MSE_total.append(MSE2)
        MSE_total.append(MSE3)
        MSE_total.append(MSE4)
        MSE_total.append(MSE5)
        MSE_total.append(MSE6)
        return MSE_total, weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4, weight_NP5, weight_NP6

    @staticmethod
    def selection(iteration, MSE_total, weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4, weight_NP5,
                  weight_NP6):
        MSE_top = []
        MSE_place = []
        # отбор: поиск лучшего MSE

        for k in range(3):
            # print(MSE_total)
            # print(range(len(MSE_total)))

            for j in range(len(MSE_total)):
                if MSE_total[j] == min(MSE_total):
                    # print(j, " - ", MSE_total[j])
                    MSE_place.append(j)
                    MSE_top.append(MSE_total[j])
                    MSE_total[j] = 10000.0
                    break
        # print(MSE_top)
        # print(MSE_place)


        # отбор: вывод лучших весов по лучшим MSE
        weight_top = []
        for numberMSE_place in range(3):
            # for numberWeight in range(7):
            if MSE_place[numberMSE_place] == 0:
                # print("weight_NP0", weight_NP0)
                weight_top.append(weight_NP0.tolist())
            if MSE_place[numberMSE_place] == 1:
                # print("weight_NP1", weight_NP1)
                weight_top.append(weight_NP1.tolist())
            if MSE_place[numberMSE_place] == 2:
                # print("weight_NP2", weight_NP2)
                weight_top.append(weight_NP2.tolist())
            if MSE_place[numberMSE_place] == 3:
                # print("weight_NP3", weight_NP3)
                weight_top.append(weight_NP3.tolist())
            if MSE_place[numberMSE_place] == 4:
                # print("weight_NP4", weight_NP4)
                weight_top.append(weight_NP4.tolist())
            if MSE_place[numberMSE_place] == 5:
                # print("weight_NP5", weight_NP5)
                weight_top.append(weight_NP5.tolist())
            if MSE_place[numberMSE_place] == 6:
                # print("weight_NP6", weight_NP6)
                weight_top.append(weight_NP6.tolist())
        # print(weight_top, "\n")
        # print(weight_top[0])
        # print("------------------------------")
        table = [[iteration, MSE_top[0], MSE_top[1], MSE_top[2], MSE_total[0], MSE_total[1], MSE_total[2], MSE_total[3],
                  MSE_total[4], MSE_total[5], MSE_total[6]]]
        print(tabulate(table,
                       # headers=["MSE_top[0]", "MSE_top[1]",
                       #          "MSE_top[2]", "kLearningRate",
                       #          "current alphaLearningRate",
                       #          "epsilonLimitation", "errorAvg(MSE)",
                       #          "MSE[i]", "MSE[i-1]", "MSE[0]", "abs(MSE[i] - MSE[i-1])",
                       #          "weight_NP[1]", "weight_NP[2]", "abs(weight_NP[1] - weight_NP[2]"],
                       tablefmt='orgtbl'))
        return weight_top

    @staticmethod
    def crossing(weight_top):
        weight_NP0 = np.asarray(weight_top[0])
        # print("weight_NP0", weight_NP0)
        # w1 [0]
        weight_NP1_1 = np.asarray(weight_top[0])
        weight_NP1_2 = np.asarray(weight_top[1])
        weight_NP1_1List = weight_NP1_1.tolist()
        weight_NP1_2List = weight_NP1_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP1_1List[0][0]))
        string1 = str(float(weight_NP1_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w1 [1]
        string0 = str(float(weight_NP1_1List[1][0]))
        string1 = str(float(weight_NP1_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w1 [2]
        string0 = str(float(weight_NP1_1List[2][0]))
        string1 = str(float(weight_NP1_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        weight_NP1 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP1", weight_NP1)

        # w2 [0]
        weight_NP2_1 = np.asarray(weight_top[1])
        weight_NP2_2 = np.asarray(weight_top[0])
        weight_NP2_1List = weight_NP2_1.tolist()
        weight_NP2_2List = weight_NP2_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP2_1List[0][0]))
        string1 = str(float(weight_NP2_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w2 [1]
        string0 = str(float(weight_NP2_1List[1][0]))
        string1 = str(float(weight_NP2_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w2 [2]
        string0 = str(float(weight_NP2_1List[2][0]))
        string1 = str(float(weight_NP2_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        weight_NP2 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP2", weight_NP2)

        # w3 [0]
        weight_NP3_1 = np.asarray(weight_top[0])
        weight_NP3_2 = np.asarray(weight_top[2])
        weight_NP3_1List = weight_NP3_1.tolist()
        weight_NP3_2List = weight_NP3_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP3_1List[0][0]))
        string1 = str(float(weight_NP3_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w3 [1]
        string0 = str(float(weight_NP3_1List[1][0]))
        string1 = str(float(weight_NP3_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w3 [2]
        string0 = str(float(weight_NP3_1List[2][0]))
        string1 = str(float(weight_NP3_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        weight_NP3 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP3", weight_NP3)

        # w4 [0]
        weight_NP4_1 = np.asarray(weight_top[2])
        weight_NP4_2 = np.asarray(weight_top[0])
        weight_NP4_1List = weight_NP4_1.tolist()
        weight_NP4_2List = weight_NP4_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP4_1List[0][0]))
        string1 = str(float(weight_NP4_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w4 [1]
        string0 = str(float(weight_NP4_1List[1][0]))
        string1 = str(float(weight_NP4_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w4 [2]
        string0 = str(float(weight_NP4_1List[2][0]))
        string1 = str(float(weight_NP4_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        weight_NP4 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP4", weight_NP4)

        # w5 [0]
        weight_NP5_1 = np.asarray(weight_top[2])
        weight_NP5_2 = np.asarray(weight_top[1])
        weight_NP5_1List = weight_NP5_1.tolist()
        weight_NP5_2List = weight_NP5_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP5_1List[0][0]))
        string1 = str(float(weight_NP5_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w5 [1]
        string0 = str(float(weight_NP5_1List[1][0]))
        string1 = str(float(weight_NP5_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w5 [2]
        string0 = str(float(weight_NP5_1List[2][0]))
        string1 = str(float(weight_NP5_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        weight_NP5 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP5", weight_NP5)

        # w6 [0]
        weight_NP6_1 = np.asarray(weight_top[1])
        weight_NP6_2 = np.asarray(weight_top[2])
        weight_NP6_1List = weight_NP6_1.tolist()
        weight_NP6_2List = weight_NP6_2.tolist()
        # print("weight_NP0List", weight_NP1List)
        # print("weight_NP0List[0]", weight_NP1List[0])
        # print("weight_NP0List[0][0]", weight_NP1List[0][0])
        # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
        # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
        # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
        string0 = str(float(weight_NP6_1List[0][0]))
        string1 = str(float(weight_NP6_2List[0][0]))
        string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w6 [1]
        string0 = str(float(weight_NP6_1List[1][0]))
        string1 = str(float(weight_NP6_2List[1][0]))
        string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        # w6 [2]
        string0 = str(float(weight_NP6_1List[2][0]))
        string1 = str(float(weight_NP6_2List[2][0]))
        string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
                   + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
        weight_NP6 = np.vstack((float(string00), float(string10), float(string20)))
        # print("weight_NP6", weight_NP6)
        return weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4, weight_NP5, weight_NP6
