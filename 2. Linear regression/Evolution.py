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
    def startEvolution(numberOfIteration, writeToOutputTxt, mutationProbabilityVar):
        """Метод запуска эволюции в цикле.

        Args:
            numberOfIteration: число, количество шагов эволюции.
            writeToOutputTxt: флаг записи в таблицы в файл outputEvolution.txt: 1 - включен.
            mutationProbabilityVar: вероятность мутации особи.

        Returns:
            MSE_histTop0_NP.tolist(): лист, первая лучшая ошибка.
            MSE_histTop1_NP.tolist(): лист, вторая лучшая ошибка.
            MSE_histTop2_NP.tolist(): лист, третья лучшая ошибка.
            MSE_hist0_NP.tolist(): лист, изменение ошибки особи 0.
            MSE_hist1_NP.tolist(): лист, изменение ошибки особи 1.
            MSE_hist2_NP.tolist(): лист, изменение ошибки особи 2.
            MSE_hist3_NP.tolist(): лист, изменение ошибки особи 3.
            MSE_hist4_NP.tolist(): лист, изменение ошибки особи 4.
            MSE_hist5_NP.tolist(): лист, изменение ошибки особи 5.
            MSE_hist6_NP.tolist(): лист, изменение ошибки особи 6.
            wTop_NP.tolist(): лист, содержащий наилучшие веса w0 для x0, w1 для x1, w2 для x2.
        """
        weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4, weight_NP5, weight_NP6 = \
        Evolution.weightStartIntialization()

        MSE_histTop0_NP = np.zeros(numberOfIteration)
        MSE_histTop1_NP = np.zeros(numberOfIteration)
        MSE_histTop2_NP = np.zeros(numberOfIteration)

        MSE_hist0_NP = np.zeros(numberOfIteration)
        MSE_hist1_NP = np.zeros(numberOfIteration)
        MSE_hist2_NP = np.zeros(numberOfIteration)
        MSE_hist3_NP = np.zeros(numberOfIteration)
        MSE_hist4_NP = np.zeros(numberOfIteration)
        MSE_hist5_NP = np.zeros(numberOfIteration)
        MSE_hist6_NP = np.zeros(numberOfIteration)

        weight_hist01_NP = np.zeros(numberOfIteration)
        weight_hist02_NP = np.zeros(numberOfIteration)
        weight_hist03_NP = np.zeros(numberOfIteration)
        for i in range(numberOfIteration):
            MSE_top, MSE_toOutput, weight_NP0new, weight_NP1new, weight_NP2new, weight_NP3new, weight_NP4new, \
            weight_NP5new, weight_NP6new = \
                Evolution.testEvo(i, writeToOutputTxt, weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4,
                                  weight_NP5, weight_NP6, mutationProbabilityVar)
            MSE_histTop0_NP[i] = MSE_top[0]
            MSE_histTop1_NP[i] = MSE_top[1]
            MSE_histTop2_NP[i] = MSE_top[2]

            MSE_hist0_NP[i] = MSE_toOutput[0]
            MSE_hist1_NP[i] = MSE_toOutput[1]
            MSE_hist2_NP[i] = MSE_toOutput[2]
            MSE_hist3_NP[i] = MSE_toOutput[3]
            MSE_hist4_NP[i] = MSE_toOutput[4]
            MSE_hist5_NP[i] = MSE_toOutput[5]
            MSE_hist6_NP[i] = MSE_toOutput[6]

            weight_hist01_NP[i] = weight_NP0[0]
            weight_hist02_NP[i] = weight_NP0[1]
            weight_hist03_NP[i] = weight_NP0[2]

            weight_NP0 = weight_NP0new
            weight_NP1 = weight_NP1new
            weight_NP2 = weight_NP2new
            weight_NP3 = weight_NP3new
            weight_NP4 = weight_NP4new
            weight_NP5 = weight_NP5new
            weight_NP6 = weight_NP6new

        print("i(MSE_min)=", np.argmin(MSE_hist0_NP), "MSE_min=", min(MSE_hist0_NP),
              "weight0_min=", weight_hist01_NP[np.argmin(MSE_hist0_NP)],
              "weight1_min=", weight_hist02_NP[np.argmin(MSE_hist0_NP)],
              "weight2_min=", weight_hist03_NP[np.argmin(MSE_hist0_NP)])

        # wLast
        wTop_NP = []
        wTop_NP = np.vstack((float(weight_hist01_NP[np.argmin(MSE_hist0_NP)]),
                              float(weight_hist02_NP[np.argmin(MSE_hist0_NP)]),
                              float(weight_hist03_NP[np.argmin(MSE_hist0_NP)])))

        return MSE_histTop0_NP.tolist(), MSE_histTop1_NP.tolist(), MSE_histTop2_NP.tolist(), MSE_hist0_NP.tolist(), \
               MSE_hist1_NP.tolist(), MSE_hist2_NP.tolist(), MSE_hist3_NP.tolist(), MSE_hist4_NP.tolist(), \
               MSE_hist5_NP.tolist(), MSE_hist6_NP.tolist(), wTop_NP.tolist()

    #TODO: area of visibility
    # @staticmethod
    # def calculateNewPrice(wTop_NP):
    #     """Метод подсчета новых цен на основе результатов работы алгоритма.
    #
    #     Args:
    #         wTop_NP: массив numpy, содержащий наилучшие веса w0 для x0, w1 для x1, w2 для x2.
    #
    #     Returns:
    #         YNew_NP: массив numpy, содержащий рассчитанные цены.
    #     """
    #     data = DatasetProcessing.getDataset('dataset.txt')
    #     normalizeData = DatasetProcessing.getNormalizeDataset(data)
    #     area, rooms, Y = DatasetProcessing.getSeparetedData(normalizeData)
    #     x0 = np.ones(len(area))
    #     XTranspose_NP = np.vstack((x0, area, rooms))  # двумерный массив [,.....,],[,.....,]
    #     X_NP = np.transpose(XTranspose_NP)
    #     YNew_NP = X_NP.dot(wTop_NP)
    #     return YNew_NP

    @staticmethod
    def weightStartIntialization():
        """Метод инициализации начальной популяции.

        Args:

        Returns:
            weight_NP0: массив numpy, особь 0.
            weight_NP1: массив numpy, особь 1.
            weight_NP2: массив numpy, особь 2.
            weight_NP3: массив numpy, особь 3.
            weight_NP4: массив numpy, особь 4.
            weight_NP5: массив numpy, особь 5.
            weight_NP6: массив numpy, особь 6.
        """
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
        """Метод подсчета среднеквадратичной ошибки.

        Args:
            weight_NP0: массив numpy, особь 0.
            weight_NP1: массив numpy, особь 1.
            weight_NP2: массив numpy, особь 2.
            weight_NP3: массив numpy, особь 3.
            weight_NP4: массив numpy, особь 4.
            weight_NP5: массив numpy, особь 5.
            weight_NP6: массив numpy, особь 6.
        Returns:
            MSE_toProcess: массив numpy, ошибки всех особей на данной иттерации, для дальнейшей обработки.
            MSE_toOutput: массив numpy, ошибки всех особей на данной иттерации, для конечного вывода.
        """
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
        # i = 0
        YNew_NP0 = X_NP.dot(weight_NP0)
        YNew_NP1 = X_NP.dot(weight_NP1)
        YNew_NP2 = X_NP.dot(weight_NP2)
        YNew_NP3 = X_NP.dot(weight_NP3)
        YNew_NP4 = X_NP.dot(weight_NP4)
        YNew_NP5 = X_NP.dot(weight_NP5)
        YNew_NP6 = X_NP.dot(weight_NP6)

        MSE0 = np.sum((YNew_NP0 - Y_NP) ** 2) / (2 * N)
        MSE1 = np.sum((YNew_NP1 - Y_NP) ** 2) / (2 * N)
        MSE2 = np.sum((YNew_NP2 - Y_NP) ** 2) / (2 * N)
        MSE3 = np.sum((YNew_NP3 - Y_NP) ** 2) / (2 * N)
        MSE4 = np.sum((YNew_NP4 - Y_NP) ** 2) / (2 * N)
        MSE5 = np.sum((YNew_NP5 - Y_NP) ** 2) / (2 * N)
        MSE6 = np.sum((YNew_NP6 - Y_NP) ** 2) / (2 * N)

        MSE_toProcess = []
        MSE_toProcess.append(MSE0)
        MSE_toProcess.append(MSE1)
        MSE_toProcess.append(MSE2)
        MSE_toProcess.append(MSE3)
        MSE_toProcess.append(MSE4)
        MSE_toProcess.append(MSE5)
        MSE_toProcess.append(MSE6)

        MSE_toOutput = []
        MSE_toOutput.append(MSE0)
        MSE_toOutput.append(MSE1)
        MSE_toOutput.append(MSE2)
        MSE_toOutput.append(MSE3)
        MSE_toOutput.append(MSE4)
        MSE_toOutput.append(MSE5)
        MSE_toOutput.append(MSE6)
        return MSE_toProcess, MSE_toOutput

    @staticmethod
    def mutationProbability(mutationProbabilityVar):
        """Метод подсчета вероятности мутации особи и её хромосом.

        Args:
            mutationProbabilityVar: вероятность мутации особи.

        Returns:
            mutationIndividual: число, вероятность мутации особи.
            mutationChromosomes: число, вероятность 33% мутации одной из трех хромосом.
        """
        mutationIndividual = random.randint(1, 100)  # mutationProbabilityVar %
        if (mutationIndividual <= mutationProbabilityVar):
            mutationChromosomes = random.randint(1, 99)  # 33%
        else:
            mutationChromosomes = 0
        return mutationIndividual, mutationChromosomes

    @staticmethod
    def selectionIndividuals(parentFirst, parentSecond, weight, mutationProbabilityVar):
        """Метод проведения селекции особей.

        Args:
            parentFirst: число, первый родитель.
            parentSecond: число, второй родитель.
            weight: массив numpy, хромосомы родителей.

        Returns:
            w_NPnew: массив numpy, новая особь.
        """
        weight_NP1_1 = np.asarray(weight[parentFirst])
        weight_NP1_2 = np.asarray(weight[parentSecond])
        weight_NP1_1List = weight_NP1_1.tolist()
        weight_NP1_2List = weight_NP1_2.tolist()
        string10 = str(float(weight_NP1_1List[0][0]))
        string20 = str(float(weight_NP1_2List[0][0]))
        string11 = str(float(weight_NP1_1List[1][0]))
        string21 = str(float(weight_NP1_2List[1][0]))
        string12 = str(float(weight_NP1_1List[2][0]))
        string22 = str(float(weight_NP1_2List[2][0]))

        # mutation
        mutationIndividual, mutationChromosomes = Evolution.mutationProbability(mutationProbabilityVar)
        if (mutationIndividual < 5):
            weight_mutation = np.random.randn(3, 1)
            if (mutationChromosomes < 33):
                string10 = str(float(weight_mutation[0][0]))
            elif (33 <= mutationChromosomes < 66):
                string11 = str(float(weight_mutation[1][0]))
            elif (99 <= mutationChromosomes):
                string12 = str(float(weight_mutation[2][0]))
        # w[0]
        string0 = string10[:5] + string20[5] + string10[6] + string20[7] + string10[8] + string20[9] + \
                   string10[10] + string20[11] + string10[12:]
        # w[1]
        string1 = string11[:5] + string21[5] + string11[6] + string21[7] + string11[8] + string21[9] + \
                   string11[10] + string21[11] + string11[12:]
        # w[2]
        string2 = string12[:5] + string22[5] + string12[6] + string22[7] + string12[8] + string22[9] + \
                   string12[10] + string22[11] + string12[12:]
        w_NPnew = np.vstack((float(string0), float(string1), float(string2)))
        return w_NPnew

    @staticmethod
    def testEvo(evolutionIteration, writeToOutputTxt, weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4,
                weight_NP5, weight_NP6, mutationProbabilityVar):
        """Метод проведения эволюции.

        Args:
            evolutionIteration: число, текущая итерация эволюции.
            writeToOutputTxt: флаг записи в таблицы в файл outputEvolution.txt: 1 - включен.
            weight_NP0: массив numpy, особь 0.
            weight_NP1: массив numpy, особь 1.
            weight_NP2: массив numpy, особь 2.
            weight_NP3: массив numpy, особь 3.
            weight_NP4: массив numpy, особь 4.
            weight_NP5: массив numpy, особь 5.
            weight_NP6: массив numpy, особь 6.
            mutationProbabilityVar: вероятность мутации особи.

        Returns:
            MSE_toз: массив numpy, ошибки всех особей на данной иттерации, для дальнейшей обработки.
            MSE_toOutput: массив numpy, ошибки всех особей на данной иттерации, для конечного вывода.
            weight_NP0new: массив numpy, обновленный, особь 0.
            weight_NP1new: массив numpy, обновленный, особь 1.
            weight_NP2new: массив numpy, обновленный, особь 2.
            weight_NP3new: массив numpy, обновленный, особь 3.
            weight_NP4new: массив numpy, обновленный, особь 4.
            weight_NP5new: массив numpy, обновленный, особь 5.
            weight_NP6new: массив numpy, обновленный, особь 6.
        """
        MSE_toProcess, MSE_toOutput = \
            Evolution.calculateMSE(weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4, weight_NP5, weight_NP6)
        MSE_top = []
        MSE_place = []
        # отбор: поиск лучшего MSE
        for k in range(3):
            for j in range(len(MSE_toProcess)):
                if MSE_toProcess[j] == min(MSE_toProcess):
                    # print(j, " - ", MSE_total[j])
                    MSE_place.append(j)
                    MSE_top.append(MSE_toProcess[j])
                    MSE_toProcess[j] = 10000.0
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

        weight_NP0 = np.asarray(weight_top[0])
        weight_NP0List = weight_NP0.tolist()
        if (writeToOutputTxt == 1):
            my_file = open('outputEvolution.txt', 'a')
            table = [[evolutionIteration, MSE_top[0], float(weight_NP0List[0][0]), float(weight_NP0List[1][0]),
                    float(weight_NP0List[2][0]), MSE_top[1], MSE_top[2], MSE_toOutput[0], MSE_toOutput[1],
                      MSE_toOutput[2], MSE_toOutput[3], MSE_toOutput[4], MSE_toOutput[5], MSE_toOutput[6]]]
            if (evolutionIteration == 0):
                my_file.write(tabulate(table,
                               headers=["i", "MSE_top[0]", "w0", "w1", "w2", "MSE_top[1]",
                                        "MSE_top[2]", "MSE_total[0]", "MSE_total[1]", "MSE_total[2]",
                                        "MSE_total[3]", "MSE_total[4]", "MSE_total[5]", "MSE_total[6]"],
                               tablefmt='orgtbl'))
            else:
                my_file.write(tabulate(table,
                                       # headers=["i", "MSE_top[0]", "w0", "w1", "w2", "MSE_top[1]",
                                       #          "MSE_top[2]", "MSE_total[0]", "MSE_total[1]", "MSE_total[2]",
                                       #          "MSE_total[3]", "MSE_total[4]", "MSE_total[5]", "MSE_total[6]"],
                                       tablefmt='orgtbl'))
            my_file.write("\n")
            my_file.close()
        if (evolutionIteration == 0):
            print(tabulate(table,
                               headers=["i", "MSE_top[0]", "w0", "w1", "w2", "MSE_top[1]",
                                        "MSE_top[2]", "MSE_total[0]", "MSE_total[1]", "MSE_total[2]",
                                        "MSE_total[3]", "MSE_total[4]", "MSE_total[5]", "MSE_total[6]"],
                               tablefmt='orgtbl'))
        else:
            print(tabulate(table,
                               # headers=["i", "MSE_top[0]", "w0", "w1", "w2", "MSE_top[1]",
                               #          "MSE_top[2]", "MSE_total[0]", "MSE_total[1]", "MSE_total[2]",
                               #          "MSE_total[3]", "MSE_total[4]", "MSE_total[5]", "MSE_total[6]"],
                               tablefmt='orgtbl'))

        # new weight calculate
        # w0
        weight_NP0new = np.asarray(weight_top[0])
        # w1
        weight_NP1new = Evolution.selectionIndividuals(0, 1, weight_top, mutationProbabilityVar)
        # w2
        weight_NP2new = Evolution.selectionIndividuals(1, 0, weight_top, mutationProbabilityVar)
        # w3
        weight_NP3new = Evolution.selectionIndividuals(0, 2, weight_top, mutationProbabilityVar)
        # w4
        weight_NP4new = Evolution.selectionIndividuals(2, 0, weight_top, mutationProbabilityVar)
        # w5
        weight_NP5new = Evolution.selectionIndividuals(2, 1, weight_top, mutationProbabilityVar)
        # w6
        weight_NP6new = Evolution.selectionIndividuals(1, 2, weight_top, mutationProbabilityVar)

        return MSE_top, MSE_toOutput, weight_NP0new, weight_NP1new, weight_NP2new, \
               weight_NP3new, weight_NP4new, weight_NP5new, weight_NP6new

    # @staticmethod
    # def selection(iteration, MSE_total, weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4, weight_NP5,
    #               weight_NP6):
    #     MSE_top = []
    #     MSE_place = []
    #     # отбор: поиск лучшего MSE
    #
    #     for k in range(3):
    #         # print(MSE_total)
    #         # print(range(len(MSE_total)))
    #
    #         for j in range(len(MSE_total)):
    #             if MSE_total[j] == min(MSE_total):
    #                 # print(j, " - ", MSE_total[j])
    #                 MSE_place.append(j)
    #                 MSE_top.append(MSE_total[j])
    #                 MSE_total[j] = 10000.0
    #                 break
    #     # print(MSE_top)
    #     # print(MSE_place)
    #
    #
    #     # отбор: вывод лучших весов по лучшим MSE
    #     weight_top = []
    #     for numberMSE_place in range(3):
    #         # for numberWeight in range(7):
    #         if MSE_place[numberMSE_place] == 0:
    #             # print("weight_NP0", weight_NP0)
    #             weight_top.append(weight_NP0.tolist())
    #         if MSE_place[numberMSE_place] == 1:
    #             # print("weight_NP1", weight_NP1)
    #             weight_top.append(weight_NP1.tolist())
    #         if MSE_place[numberMSE_place] == 2:
    #             # print("weight_NP2", weight_NP2)
    #             weight_top.append(weight_NP2.tolist())
    #         if MSE_place[numberMSE_place] == 3:
    #             # print("weight_NP3", weight_NP3)
    #             weight_top.append(weight_NP3.tolist())
    #         if MSE_place[numberMSE_place] == 4:
    #             # print("weight_NP4", weight_NP4)
    #             weight_top.append(weight_NP4.tolist())
    #         if MSE_place[numberMSE_place] == 5:
    #             # print("weight_NP5", weight_NP5)
    #             weight_top.append(weight_NP5.tolist())
    #         if MSE_place[numberMSE_place] == 6:
    #             # print("weight_NP6", weight_NP6)
    #             weight_top.append(weight_NP6.tolist())
    #     # print(weight_top, "\n")
    #     # print(weight_top[0])
    #     # print("------------------------------")
    #     table = [[iteration, MSE_top[0], MSE_top[1], MSE_top[2], MSE_total[0], MSE_total[1], MSE_total[2], MSE_total[3],
    #               MSE_total[4], MSE_total[5], MSE_total[6]]]
    #     print(tabulate(table,
    #                    # headers=["MSE_top[0]", "MSE_top[1]",
    #                    #          "MSE_top[2]", "kLearningRate",
    #                    #          "current alphaLearningRate",
    #                    #          "epsilonLimitation", "errorAvg(MSE)",
    #                    #          "MSE[i]", "MSE[i-1]", "MSE[0]", "abs(MSE[i] - MSE[i-1])",
    #                    #          "weight_NP[1]", "weight_NP[2]", "abs(weight_NP[1] - weight_NP[2]"],
    #                    tablefmt='orgtbl'))
    #     return weight_top
    #
    # @staticmethod
    # def crossing(weight_top):
    #     weight_NP0 = np.asarray(weight_top[0])
    #     # print("weight_NP0", weight_NP0)
    #     # w1 [0]
    #     weight_NP1_1 = np.asarray(weight_top[0])
    #     weight_NP1_2 = np.asarray(weight_top[1])
    #     weight_NP1_1List = weight_NP1_1.tolist()
    #     weight_NP1_2List = weight_NP1_2.tolist()
    #     # print("weight_NP0List", weight_NP1List)
    #     # print("weight_NP0List[0]", weight_NP1List[0])
    #     # print("weight_NP0List[0][0]", weight_NP1List[0][0])
    #     # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
    #     # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
    #     # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
    #     string0 = str(float(weight_NP1_1List[0][0]))
    #     string1 = str(float(weight_NP1_2List[0][0]))
    #     string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w1 [1]
    #     string0 = str(float(weight_NP1_1List[1][0]))
    #     string1 = str(float(weight_NP1_2List[1][0]))
    #     string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w1 [2]
    #     string0 = str(float(weight_NP1_1List[2][0]))
    #     string1 = str(float(weight_NP1_2List[2][0]))
    #     string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     weight_NP1 = np.vstack((float(string00), float(string10), float(string20)))
    #     # print("weight_NP1", weight_NP1)
    #
    #     # w2 [0]
    #     weight_NP2_1 = np.asarray(weight_top[1])
    #     weight_NP2_2 = np.asarray(weight_top[0])
    #     weight_NP2_1List = weight_NP2_1.tolist()
    #     weight_NP2_2List = weight_NP2_2.tolist()
    #     # print("weight_NP0List", weight_NP1List)
    #     # print("weight_NP0List[0]", weight_NP1List[0])
    #     # print("weight_NP0List[0][0]", weight_NP1List[0][0])
    #     # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
    #     # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
    #     # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
    #     string0 = str(float(weight_NP2_1List[0][0]))
    #     string1 = str(float(weight_NP2_2List[0][0]))
    #     string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w2 [1]
    #     string0 = str(float(weight_NP2_1List[1][0]))
    #     string1 = str(float(weight_NP2_2List[1][0]))
    #     string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w2 [2]
    #     string0 = str(float(weight_NP2_1List[2][0]))
    #     string1 = str(float(weight_NP2_2List[2][0]))
    #     string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10] \
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     weight_NP2 = np.vstack((float(string00), float(string10), float(string20)))
    #     # print("weight_NP2", weight_NP2)
    #
    #     # w3 [0]
    #     weight_NP3_1 = np.asarray(weight_top[0])
    #     weight_NP3_2 = np.asarray(weight_top[2])
    #     weight_NP3_1List = weight_NP3_1.tolist()
    #     weight_NP3_2List = weight_NP3_2.tolist()
    #     # print("weight_NP0List", weight_NP1List)
    #     # print("weight_NP0List[0]", weight_NP1List[0])
    #     # print("weight_NP0List[0][0]", weight_NP1List[0][0])
    #     # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
    #     # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
    #     # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
    #     string0 = str(float(weight_NP3_1List[0][0]))
    #     string1 = str(float(weight_NP3_2List[0][0]))
    #     string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w3 [1]
    #     string0 = str(float(weight_NP3_1List[1][0]))
    #     string1 = str(float(weight_NP3_2List[1][0]))
    #     string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w3 [2]
    #     string0 = str(float(weight_NP3_1List[2][0]))
    #     string1 = str(float(weight_NP3_2List[2][0]))
    #     string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     weight_NP3 = np.vstack((float(string00), float(string10), float(string20)))
    #     # print("weight_NP3", weight_NP3)
    #
    #     # w4 [0]
    #     weight_NP4_1 = np.asarray(weight_top[2])
    #     weight_NP4_2 = np.asarray(weight_top[0])
    #     weight_NP4_1List = weight_NP4_1.tolist()
    #     weight_NP4_2List = weight_NP4_2.tolist()
    #     # print("weight_NP0List", weight_NP1List)
    #     # print("weight_NP0List[0]", weight_NP1List[0])
    #     # print("weight_NP0List[0][0]", weight_NP1List[0][0])
    #     # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
    #     # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
    #     # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
    #     string0 = str(float(weight_NP4_1List[0][0]))
    #     string1 = str(float(weight_NP4_2List[0][0]))
    #     string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w4 [1]
    #     string0 = str(float(weight_NP4_1List[1][0]))
    #     string1 = str(float(weight_NP4_2List[1][0]))
    #     string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w4 [2]
    #     string0 = str(float(weight_NP4_1List[2][0]))
    #     string1 = str(float(weight_NP4_2List[2][0]))
    #     string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     weight_NP4 = np.vstack((float(string00), float(string10), float(string20)))
    #     # print("weight_NP4", weight_NP4)
    #
    #     # w5 [0]
    #     weight_NP5_1 = np.asarray(weight_top[2])
    #     weight_NP5_2 = np.asarray(weight_top[1])
    #     weight_NP5_1List = weight_NP5_1.tolist()
    #     weight_NP5_2List = weight_NP5_2.tolist()
    #     # print("weight_NP0List", weight_NP1List)
    #     # print("weight_NP0List[0]", weight_NP1List[0])
    #     # print("weight_NP0List[0][0]", weight_NP1List[0][0])
    #     # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
    #     # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
    #     # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
    #     string0 = str(float(weight_NP5_1List[0][0]))
    #     string1 = str(float(weight_NP5_2List[0][0]))
    #     string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w5 [1]
    #     string0 = str(float(weight_NP5_1List[1][0]))
    #     string1 = str(float(weight_NP5_2List[1][0]))
    #     string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w5 [2]
    #     string0 = str(float(weight_NP5_1List[2][0]))
    #     string1 = str(float(weight_NP5_2List[2][0]))
    #     string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     weight_NP5 = np.vstack((float(string00), float(string10), float(string20)))
    #     # print("weight_NP5", weight_NP5)
    #
    #     # w6 [0]
    #     weight_NP6_1 = np.asarray(weight_top[1])
    #     weight_NP6_2 = np.asarray(weight_top[2])
    #     weight_NP6_1List = weight_NP6_1.tolist()
    #     weight_NP6_2List = weight_NP6_2.tolist()
    #     # print("weight_NP0List", weight_NP1List)
    #     # print("weight_NP0List[0]", weight_NP1List[0])
    #     # print("weight_NP0List[0][0]", weight_NP1List[0][0])
    #     # print("type(weight_NP0List[0][0])", type(weight_NP1List[0][0]))
    #     # print("str(float(weight_NP0List[0][0]))", str(float(weight_NP1List[0][0])))
    #     # print("type(str(float(weight_NP0List[0][0])))", type(str(float(weight_NP1List[0][0]))))
    #     string0 = str(float(weight_NP6_1List[0][0]))
    #     string1 = str(float(weight_NP6_2List[0][0]))
    #     string00 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w6 [1]
    #     string0 = str(float(weight_NP6_1List[1][0]))
    #     string1 = str(float(weight_NP6_2List[1][0]))
    #     string10 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     # w6 [2]
    #     string0 = str(float(weight_NP6_1List[2][0]))
    #     string1 = str(float(weight_NP6_2List[2][0]))
    #     string20 = string0[:5] + string1[5] + string0[6] + string1[7] + string0[8] + string1[9] + string0[10]\
    #                + string1[11] + string0[12] + string1[13] + string0[14] + string1[15]
    #     weight_NP6 = np.vstack((float(string00), float(string10), float(string20)))
    #     # print("weight_NP6", weight_NP6)
    #     return weight_NP0, weight_NP1, weight_NP2, weight_NP3, weight_NP4, weight_NP5, weight_NP6
