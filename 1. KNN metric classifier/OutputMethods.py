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
        k = 15
        data_1, trainDots_1, testDots_1 = DatasetProcessing.getDataset(tVAR, "none")
        data_2, trainDots_2, testDots_2 = DatasetProcessing.getDataset(tVAR, "none")
        data_3, trainDots_3, testDots_3 = DatasetProcessing.getDataset(tVAR, "none")
        data_4, trainDots_4, testDots_4 = DatasetProcessing.getDataset(tVAR, "none")
        data_5, trainDots_5, testDots_5 = DatasetProcessing.getDataset(tVAR, "none")
        data_6, trainDots_6, testDots_6 = DatasetProcessing.getDataset(tVAR, "none")
        data_11, trainDots_11, testDots_11 = DatasetProcessing.getDataset(tVAR, "elliptic")
        data_12, trainDots_12, testDots_12 = DatasetProcessing.getDataset(tVAR, "elliptic")
        data_13, trainDots_13, testDots_13 = DatasetProcessing.getDataset(tVAR, "elliptic")
        data_14, trainDots_14, testDots_14 = DatasetProcessing.getDataset(tVAR, "elliptic")
        data_15, trainDots_15, testDots_15 = DatasetProcessing.getDataset(tVAR, "elliptic")
        data_16, trainDots_16, testDots_16 = DatasetProcessing.getDataset(tVAR, "elliptic")
        data_21, trainDots_21, testDots_21 = DatasetProcessing.getDataset(tVAR, "hyperbolic")
        data_22, trainDots_22, testDots_22 = DatasetProcessing.getDataset(tVAR, "hyperbolic")
        data_23, trainDots_23, testDots_23 = DatasetProcessing.getDataset(tVAR, "hyperbolic")
        data_24, trainDots_24, testDots_24 = DatasetProcessing.getDataset(tVAR, "hyperbolic")
        data_25, trainDots_25, testDots_25 = DatasetProcessing.getDataset(tVAR, "hyperbolic")
        data_26, trainDots_26, testDots_26 = DatasetProcessing.getDataset(tVAR, "hyperbolic")

        # Plot.buildPlotWithAllDots(trainDots, testDots)
        testClasses_1 = [testDots_1[i][1] for i in range(len(testDots_1))]
        testClasses_2 = [testDots_2[i][1] for i in range(len(testDots_2))]
        testClasses_3 = [testDots_3[i][1] for i in range(len(testDots_3))]
        testClasses_4 = [testDots_4[i][1] for i in range(len(testDots_4))]
        testClasses_5 = [testDots_5[i][1] for i in range(len(testDots_5))]
        testClasses_6 = [testDots_6[i][1] for i in range(len(testDots_6))]
        testClasses_11 = [testDots_11[i][1] for i in range(len(testDots_11))]
        testClasses_12 = [testDots_12[i][1] for i in range(len(testDots_12))]
        testClasses_13 = [testDots_13[i][1] for i in range(len(testDots_13))]
        testClasses_14 = [testDots_14[i][1] for i in range(len(testDots_14))]
        testClasses_15 = [testDots_15[i][1] for i in range(len(testDots_15))]
        testClasses_16 = [testDots_16[i][1] for i in range(len(testDots_16))]
        testClasses_21 = [testDots_21[i][1] for i in range(len(testDots_21))]
        testClasses_22 = [testDots_22[i][1] for i in range(len(testDots_22))]
        testClasses_23 = [testDots_23[i][1] for i in range(len(testDots_23))]
        testClasses_24 = [testDots_24[i][1] for i in range(len(testDots_24))]
        testClasses_25 = [testDots_25[i][1] for i in range(len(testDots_25))]
        testClasses_26 = [testDots_26[i][1] for i in range(len(testDots_26))]

        print("*" * 212)
        # Plot.buildPlotCircle(trainDots_1, testDots_1, 50, k, "manhattan", "none")
        # Plot.buildPlotCircle(trainDots_11, testDots_11, 50, k, "euclidean", "elliptic")

        testClassesClassified_1 = DatasetProcessing.classifyKNNCircle(trainDots_1, testDots_1, k, "none",
                                                                      "manhattan", "none")
        testClassesClassified_2 = DatasetProcessing.classifyKNNCircle(trainDots_2, testDots_2, k, "gaussian",
                                                                      "manhattan", "none")
        testClassesClassified_3 = DatasetProcessing.classifyKNNCircle(trainDots_3, testDots_3, k, "logistic",
                                                                      "manhattan", "none")
        testClassesClassified_4 = DatasetProcessing.classifyKNNCircle(trainDots_4, testDots_4, k, "none",
                                                                      "euclidean", "none")
        testClassesClassified_5 = DatasetProcessing.classifyKNNCircle(trainDots_5, testDots_5, k, "gaussian",
                                                                      "euclidean", "none")
        testClassesClassified_6 = DatasetProcessing.classifyKNNCircle(trainDots_6, testDots_6, k, "logistic",
                                                                      "euclidean", "none")
        testClassesClassified_11 = DatasetProcessing.classifyKNNCircle(trainDots_11, testDots_11, k, "none",
                                                                       "manhattan", "elliptic")
        testClassesClassified_12 = DatasetProcessing.classifyKNNCircle(trainDots_12, testDots_12, k, "gaussian",
                                                                       "manhattan", "elliptic")
        testClassesClassified_13 = DatasetProcessing.classifyKNNCircle(trainDots_13, testDots_13, k, "logistic",
                                                                       "manhattan", "elliptic")
        testClassesClassified_14 = DatasetProcessing.classifyKNNCircle(trainDots_14, testDots_14, k, "none",
                                                                       "euclidean", "elliptic")
        testClassesClassified_15 = DatasetProcessing.classifyKNNCircle(trainDots_15, testDots_15, k, "gaussian",
                                                                       "euclidean", "elliptic")
        testClassesClassified_16 = DatasetProcessing.classifyKNNCircle(trainDots_16, testDots_16, k, "logistic",
                                                                       "euclidean", "elliptic")
        testClassesClassified_21 = DatasetProcessing.classifyKNNCircle(trainDots_21, testDots_21, k, "none",
                                                                       "manhattan", "hyperbolic")
        testClassesClassified_22 = DatasetProcessing.classifyKNNCircle(trainDots_22, testDots_22, k, "gaussian",
                                                                       "manhattan", "hyperbolic")
        testClassesClassified_23 = DatasetProcessing.classifyKNNCircle(trainDots_23, testDots_23, k, "logistic",
                                                                       "manhattan", "hyperbolic")
        testClassesClassified_24 = DatasetProcessing.classifyKNNCircle(trainDots_24, testDots_24, k, "none",
                                                                       "euclidean", "hyperbolic")
        testClassesClassified_25 = DatasetProcessing.classifyKNNCircle(trainDots_25, testDots_25, k, "gaussian",
                                                                       "euclidean", "hyperbolic")
        testClassesClassified_26 = DatasetProcessing.classifyKNNCircle(trainDots_26, testDots_26, k, "logistic",
                                                                       "euclidean", "hyperbolic")

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
                        [len(data_11) - len(testDots_11), len(testDots_11), k, "none", "manhattan", "elliptic",
                         Statistic.computingF1_measure(testClassesClassified_11, testClasses_11),
                         Statistic.computingRecall(testClassesClassified_11, testClasses_11),
                         Statistic.computingSpecificity(testClassesClassified_11, testClasses_11),
                         Statistic.computingPrecision(testClassesClassified_11, testClasses_11),
                         Statistic.computingAccuracy(testClassesClassified_11, testClasses_11)],
                        [len(data_12) - len(testDots_12), len(testDots_12), k, "gaussian", "manhattan", "elliptic",
                         Statistic.computingF1_measure(testClassesClassified_12, testClasses_12),
                         Statistic.computingRecall(testClassesClassified_12, testClasses_12),
                         Statistic.computingSpecificity(testClassesClassified_12, testClasses_12),
                         Statistic.computingPrecision(testClassesClassified_12, testClasses_12),
                         Statistic.computingAccuracy(testClassesClassified_12, testClasses_12)],
                        [len(data_13) - len(testDots_13), len(testDots_13), k, "euclidean", "manhattan", "elliptic",
                         Statistic.computingF1_measure(testClassesClassified_13, testClasses_13),
                         Statistic.computingRecall(testClassesClassified_13, testClasses_13),
                         Statistic.computingSpecificity(testClassesClassified_13, testClasses_13),
                         Statistic.computingPrecision(testClassesClassified_13, testClasses_13),
                         Statistic.computingAccuracy(testClassesClassified_13, testClasses_13)],
                        [len(data_14) - len(testDots_14), len(testDots_14), k, "none", "euclidean", "elliptic",
                         Statistic.computingF1_measure(testClassesClassified_14, testClasses_14),
                         Statistic.computingRecall(testClassesClassified_14, testClasses_14),
                         Statistic.computingSpecificity(testClassesClassified_14, testClasses_14),
                         Statistic.computingPrecision(testClassesClassified_14, testClasses_14),
                         Statistic.computingAccuracy(testClassesClassified_14, testClasses_14)],
                        [len(data_15) - len(testDots_15), len(testDots_15), k, "gaussian", "euclidean", "elliptic",
                         Statistic.computingF1_measure(testClassesClassified_15, testClasses_15),
                         Statistic.computingRecall(testClassesClassified_15, testClasses_15),
                         Statistic.computingSpecificity(testClassesClassified_15, testClasses_15),
                         Statistic.computingPrecision(testClassesClassified_15, testClasses_15),
                         Statistic.computingAccuracy(testClassesClassified_15, testClasses_15)],
                        [len(data_16) - len(testDots_16), len(testDots_16), k, "euclidean", "euclidean", "elliptic",
                         Statistic.computingF1_measure(testClassesClassified_16, testClasses_16),
                         Statistic.computingRecall(testClassesClassified_16, testClasses_16),
                         Statistic.computingSpecificity(testClassesClassified_16, testClasses_16),
                         Statistic.computingPrecision(testClassesClassified_16, testClasses_16),
                         Statistic.computingAccuracy(testClassesClassified_16, testClasses_16)],
                        [len(data_21) - len(testDots_21), len(testDots_21), k, "none", "manhattan", "hyperbolic",
                         Statistic.computingF1_measure(testClassesClassified_21, testClasses_21),
                         Statistic.computingRecall(testClassesClassified_21, testClasses_21),
                         Statistic.computingSpecificity(testClassesClassified_21, testClasses_21),
                         Statistic.computingPrecision(testClassesClassified_21, testClasses_21),
                         Statistic.computingAccuracy(testClassesClassified_21, testClasses_21)],
                        [len(data_22) - len(testDots_22), len(testDots_22), k, "gaussian", "manhattan", "hyperbolic",
                         Statistic.computingF1_measure(testClassesClassified_22, testClasses_22),
                         Statistic.computingRecall(testClassesClassified_22, testClasses_22),
                         Statistic.computingSpecificity(testClassesClassified_22, testClasses_22),
                         Statistic.computingPrecision(testClassesClassified_22, testClasses_22),
                         Statistic.computingAccuracy(testClassesClassified_22, testClasses_22)],
                        [len(data_23) - len(testDots_23), len(testDots_23), k, "euclidean", "manhattan", "hyperbolic",
                         Statistic.computingF1_measure(testClassesClassified_23, testClasses_23),
                         Statistic.computingRecall(testClassesClassified_23, testClasses_23),
                         Statistic.computingSpecificity(testClassesClassified_23, testClasses_23),
                         Statistic.computingPrecision(testClassesClassified_23, testClasses_23),
                         Statistic.computingAccuracy(testClassesClassified_23, testClasses_23)],
                        [len(data_24) - len(testDots_24), len(testDots_24), k, "none", "euclidean", "hyperbolic",
                         Statistic.computingF1_measure(testClassesClassified_24, testClasses_24),
                         Statistic.computingRecall(testClassesClassified_24, testClasses_24),
                         Statistic.computingSpecificity(testClassesClassified_24, testClasses_24),
                         Statistic.computingPrecision(testClassesClassified_24, testClasses_24),
                         Statistic.computingAccuracy(testClassesClassified_24, testClasses_24)],
                        [len(data_25) - len(testDots_25), len(testDots_25), k, "gaussian", "euclidean", "hyperbolic",
                         Statistic.computingF1_measure(testClassesClassified_25, testClasses_25),
                         Statistic.computingRecall(testClassesClassified_25, testClasses_25),
                         Statistic.computingSpecificity(testClassesClassified_25, testClasses_25),
                         Statistic.computingPrecision(testClassesClassified_25, testClasses_25),
                         Statistic.computingAccuracy(testClassesClassified_25, testClasses_25)],
                        [len(data_26) - len(testDots_26), len(testDots_26), k, "euclidean", "euclidean", "hyperbolic",
                         Statistic.computingF1_measure(testClassesClassified_26, testClasses_26),
                         Statistic.computingRecall(testClassesClassified_26, testClasses_26),
                         Statistic.computingSpecificity(testClassesClassified_26, testClasses_26),
                         Statistic.computingPrecision(testClassesClassified_26, testClasses_26),
                         Statistic.computingAccuracy(testClassesClassified_26, testClasses_26)]],
                       headers=["Training dots", "Test dots", "k (neighbors)", "Kernel functions",
                                "Metrics for configuring kNN", "Spatial coordinate transformations", "F1-measure",
                                "Recall", "Specificity", "Precision", "Accuracy"], tablefmt='orgtbl'))
