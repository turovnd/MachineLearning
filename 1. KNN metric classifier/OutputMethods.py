from DatasetProcessing import *
from Plot import *
from Statistic import *
from tabulate import tabulate

"""
"""


class OutputMethods(object):
    """Initialization variables"""
    def __init__(self):
        pass

    """Метод отображения результатов
    """
    @staticmethod
    def outputTable():
        # проблема с результатами в таблице при создании одного набора переменных
        tVAR = 50
        k = 5

        data_1, trainDots_1, testDots_1 = DatasetProcessing.getDataset(tVAR)
        data_2, trainDots_2, testDots_2 = DatasetProcessing.getDataset(tVAR)
        data_3, trainDots_3, testDots_3 = DatasetProcessing.getDataset(tVAR)
        data_4, trainDots_4, testDots_4 = DatasetProcessing.getDataset(tVAR)
        data_5, trainDots_5, testDots_5 = DatasetProcessing.getDataset(tVAR)
        data_6, trainDots_6, testDots_6 = DatasetProcessing.getDataset(tVAR)
        # тестовые
        # print(len(data_1)) = 118
        # print(len(trainDots_1)) = 50
        # print(len(testDots_1)) = 68

        # Plot.buildPlotWithAllDots(trainDots, testDots)
        testClasses_1 = [testDots_1[i][1] for i in range(len(testDots_1))]
        testClasses_2 = [testDots_2[i][1] for i in range(len(testDots_2))]
        testClasses_3 = [testDots_3[i][1] for i in range(len(testDots_3))]
        testClasses_4 = [testDots_4[i][1] for i in range(len(testDots_4))]
        testClasses_5 = [testDots_5[i][1] for i in range(len(testDots_5))]
        testClasses_6 = [testDots_6[i][1] for i in range(len(testDots_6))]
        print("*" * 248)
        # Plot.buildPlotCircle(trainDots, testDots, 50, k, "manhattan")
        # Plot.buildPlotCircle(trainDots, testDots, 50, k, "euclidean")

        testClassesClassified_1 = DatasetProcessing.classifyKNNCircle(trainDots_1, testDots_1, k, "none",
                                                                      "manhattan")
        testClassesClassified_2 = DatasetProcessing.classifyKNNCircle(trainDots_2, testDots_2, k, "gaussian",
                                                                      "manhattan")
        testClassesClassified_3 = DatasetProcessing.classifyKNNCircle(trainDots_3, testDots_3, k, "logistic",
                                                                      "manhattan")

        testClassesClassified_4 = DatasetProcessing.classifyKNNCircle(trainDots_4, testDots_4, k, "none",
                                                                      "euclidean")
        testClassesClassified_5 = DatasetProcessing.classifyKNNCircle(trainDots_5, testDots_5, k, "gaussian",
                                                                      "euclidean")
        testClassesClassified_6 = DatasetProcessing.classifyKNNCircle(trainDots_6, testDots_6, k, "logistic",
                                                                      "euclidean")

        # TODO пространственные преобразования
        # arr = []
        # for i in range(len(testDots)):
        #     arr.append([[trainDots[i][0][0] * math.pi, trainDots[i][0][1]], trainDots[i][1]])
        # trainDots = arr
        #
        # print("*" * 50)
        # print("classifyKNN_circle  " + "gaussian" + "  пространственное преобразование")
        # testClassesClassified = DatasetProcessing.classifyKNNCircle(trainDots, testDots, k, "gaussian")
        # Statistic.computingF1_measure(testClassesClassified, testClasses)
        #
        # arr = []
        # for i in range(len(testDots)):
        #     arr.append([[trainDots[i][0][0] * 10 * math.pi, trainDots[i][0][1] * 10 * math.pi], trainDots[i][1]])
        # trainDots = arr
        #
        # print("*" * 50)
        # print("classifyKNN_circle  " + "logistic" + "  пространственное преобразование 2 ")
        # testClassesClassified = DatasetProcessing.classifyKNNCircle(trainDots, testDots, k, "logistic")
        # Statistic.computingF1_measure(testClassesClassified, testClasses)

        # почему то trainDots_ = 118
        print(tabulate([[len(data_1) - len(testDots_1), len(testDots_1), k, "none", "manhattan", "none",
                         Statistic.computingF1_measure(testClassesClassified_1, testClasses_1),
                         Statistic.computingRecall(testClassesClassified_1, testClasses_1),
                         Statistic.computingSpecificity(testClassesClassified_1, testClasses_1),
                         Statistic.computingPrecision(testClassesClassified_1, testClasses_1),
                         Statistic.computingAccuracy(testClassesClassified_1, testClasses_1)],
                        [len(data_2) - len(testDots_2), len(testDots_2), k,  "gaussian", "manhattan", "none",
                         Statistic.computingF1_measure(testClassesClassified_2, testClasses_2),
                         Statistic.computingRecall(testClassesClassified_2, testClasses_2),
                         Statistic.computingSpecificity(testClassesClassified_2, testClasses_2),
                         Statistic.computingPrecision(testClassesClassified_2, testClasses_2),
                         Statistic.computingAccuracy(testClassesClassified_2, testClasses_2)],
                        [len(data_3) - len(testDots_3), len(testDots_3), k,  "logistic", "manhattan", "none",
                         Statistic.computingF1_measure(testClassesClassified_3, testClasses_3),
                         Statistic.computingRecall(testClassesClassified_3, testClasses_3),
                         Statistic.computingSpecificity(testClassesClassified_3, testClasses_3),
                         Statistic.computingPrecision(testClassesClassified_3, testClasses_3),
                         Statistic.computingAccuracy(testClassesClassified_3, testClasses_3)],
                        [len(data_4) - len(testDots_4), len(testDots_4), k,  "none", "euclidean", "none",
                         Statistic.computingF1_measure(testClassesClassified_4, testClasses_4),
                         Statistic.computingRecall(testClassesClassified_4, testClasses_4),
                         Statistic.computingSpecificity(testClassesClassified_4, testClasses_4),
                         Statistic.computingPrecision(testClassesClassified_4, testClasses_4),
                         Statistic.computingAccuracy(testClassesClassified_4, testClasses_4)],
                        [len(data_5) - len(testDots_5), len(testDots_5), k,  "gaussian", "euclidean", "none",
                         Statistic.computingF1_measure(testClassesClassified_5, testClasses_5),
                         Statistic.computingRecall(testClassesClassified_5, testClasses_5),
                         Statistic.computingSpecificity(testClassesClassified_5, testClasses_5),
                         Statistic.computingPrecision(testClassesClassified_5, testClasses_5),
                         Statistic.computingAccuracy(testClassesClassified_5, testClasses_5)],
                        [len(data_6) - len(testDots_6), len(testDots_6), k,  "logistic", "euclidean", "none",
                         Statistic.computingF1_measure(testClassesClassified_6, testClasses_6),
                         Statistic.computingRecall(testClassesClassified_6, testClasses_6),
                         Statistic.computingSpecificity(testClassesClassified_6, testClasses_6),
                         Statistic.computingPrecision(testClassesClassified_6, testClasses_6),
                         Statistic.computingAccuracy(testClassesClassified_6, testClasses_6)],
                        ["", "", k, "none", "manhattan", "elliptic", "", "", "", "", ""],
                        ["", "", k, "gaussian", "manhattan", "elliptic", "", "", "", "", ""],
                        ["", "", k, "euclidean", "manhattan", "elliptic", "", "", "", "", ""],
                        ["", "", k, "none", "euclidean", "elliptic", "", "", "", "", ""],
                        ["", "", k, "gaussian", "euclidean", "elliptic", "", "", "", "", ""],
                        ["", "", k, "euclidean", "euclidean", "elliptic", "", "", "", "", ""],
                        ["", "", k, "none", "manhattan", "hyperbolic", "", "", "", "", ""],
                        ["", "", k, "gaussian", "manhattan", "hyperbolic", "", "", "", "", ""],
                        ["", "", k, "euclidean", "manhattan", "hyperbolic", "", "", "", "", ""],
                        ["", "", k, "none", "euclidean", "hyperbolic", "", "", "", "", ""],
                        ["", "", k, "gaussian", "euclidean", "hyperbolic", "", "", "", "", ""],
                        ["", "", k, "euclidean", "euclidean", "hyperbolic", "", "", "", "", ""]],
                       headers=["Training dots", "Test dots", "k (neighbors)", "Kernel functions",
                                "Metrics for configuring kNN", "Spatial coordinate transformations", "F1-measure",
                                "Recall", "Specificity", "Precision", "Accuracy"], tablefmt='orgtbl'))