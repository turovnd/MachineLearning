import random
import numpy as np
from matplotlib.colors import ListedColormap

from DatasetProcessing import *
from Plot import *
from Statistic import *

if __name__ == '__main__':
    tVAR = 50
    data, trainDots, testDots = DatasetProcessing.getDataset(tVAR)
    Plot.buildPlotWithAllDots(trainDots, testDots)
    Plot.buildPlotCentroid(trainDots, testDots, 50)
    # start
    testClasses = [testDots[i][1] for i in range(len(testDots))]
    print('testClasses', testClasses)

    print("*" * 50)
    print("classifyKNN_center")
    testClassesClassified = DatasetProcessing.classifyKNNCentroid(trainDots, testDots)
    Statistic.computingF1_measure(testClassesClassified, testClasses)

    k = 5
    core = ["gaussian", "logistic"]
    print("*" * 50)
    Plot.buildPlotCircle(trainDots, testDots, 50, k)
    print("classifyKNN_circle  without core")
    testClassesClassified = DatasetProcessing.classifyKNNCircle(trainDots, testDots, k, 0)
    Statistic.computingF1_measure(testClassesClassified, testClasses)

    print("*" * 50)
    print("classifyKNN_circle  " + core[0])
    # finish
    testClassesClassified = DatasetProcessing.classifyKNNCircle(trainDots, testDots, k, core[0])
    print('testClassesClassified', testClassesClassified)
    Statistic.computingF1_measure(testClassesClassified, testClasses)

    print("*" * 50)
    print("classifyKNN_circle  " + core[1])
    testClassesClassified = DatasetProcessing.classifyKNNCircle(trainDots, testDots, k, core[1])
    Statistic.computingF1_measure(testClassesClassified, testClasses)

    arr = []
    for i in range(len(testDots)):
        arr.append([[trainDots[i][0][0] * math.pi, trainDots[i][0][1]], trainDots[i][1]])
    trainDots = arr

    print("*" * 50)
    print("classifyKNN_circle  " + core[1] + "  пространственное преобразование")
    testClassesClassified = DatasetProcessing.classifyKNNCircle(trainDots, testDots, k, core[1])
    Statistic.computingF1_measure(testClassesClassified, testClasses)

    arr = []
    for i in range(len(testDots)):
        arr.append([[trainDots[i][0][0] * 10 * math.pi, trainDots[i][0][1] * 10 * math.pi], trainDots[i][1]])
    trainDots = arr

    print("*" * 50)
    print("classifyKNN_circle  " + core[1] + "  пространственное преобразование 2 ")
    testClassesClassified = DatasetProcessing.classifyKNNCircle(trainDots, testDots, k, core[1])
    Statistic.computingF1_measure(testClassesClassified, testClasses)
