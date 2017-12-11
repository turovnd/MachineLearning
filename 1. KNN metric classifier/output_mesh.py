# coding=utf-8
from DatasetProcessing import *
from Plot import *
from Statistic import *
from matplotlib.colors import ListedColormap
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    kernelFunctions = ["none", "gaussian", "logistic"]
    metrics = ["manhattan", "euclidean"]
    coordinateTransformations = "none"  # ["none", "elliptic", "hyperbolic"]
    number_trainDots = 50
    k_neighbors = 10
    k_fold = 9

    # Входной датасет, не изменяется в пределах одного `coordinateTransformation`
    # индекс датасето соответстует индексу `coordinateTransformations`
    data = DatasetProcessing.getDataset('dataset.txt', coordinateTransformations)

    trainDots, testDots = DatasetProcessing.getTrainTestDots(number_trainDots, data)

    metric = metrics[0]
    kFun = kernelFunctions[0]

    testClasses = [testDots[i][1] for i in range(len(testDots))]
    testClassesClassified = []

    for i in range(len(testDots)):
        dot_class = DatasetProcessing.classifyDotCircle(trainDots, testDots[i][0], k_fold, kFun, metric, coordinateTransformations)
        testDots[i][1] = dot_class
        trainDots.append(testDots[i])
        testClassesClassified.append(dot_class)

    print("Data length   = ", len(data))
    print("Training dots = ", number_trainDots)
    print("Test dots     = ", len(data) - number_trainDots)
    print("k_neighbors   = ", k_neighbors)
    print("k_fold        = ", k_fold)
    print("F1_measure    = ", Statistic.computingF1_measure(testClassesClassified, testClasses))

    def generateTestMesh(trainData):
        x_min = min([trainData[i][0][0] for i in range(len(trainData))]) - 1.0
        x_max = max([trainData[i][0][0] for i in range(len(trainData))]) + 1.0
        y_min = min([trainData[i][0][1] for i in range(len(trainData))]) - 1.0
        y_max = max([trainData[i][0][1] for i in range(len(trainData))]) + 1.0
        h = 0.05
        testX, testY = np.meshgrid(np.arange(x_min, x_max, h),
                                   np.arange(y_min, y_max, h))
        return [testX, testY]


    # Main classification procedure
    def classifyKNN (trainData, testData, k, numberOfClasses):
        testLabels = []
        for testPoint in testData:
            #Claculate distances between test point and all of the train points
            testDist = [[DatasetProcessing.computingEuclideanDistance2D(testPoint, trainData[i][0]), trainData[i][1]] for i in range(len(trainData))]
            #How many points of each class among nearest K
            stat = [0 for i in range(numberOfClasses)]
            for d in sorted(testDist)[0:k]:
                stat[d[1]] += 1
            # Assign a class with the most number of occurences among K nearest neighbours
            testLabels.append(sorted(zip(stat, range(numberOfClasses)), reverse=True)[0][1])
        return testLabels

    testMesh = generateTestMesh(trainDots)
    testMeshLabels = classifyKNN(trainDots, zip(testMesh[0].ravel(), testMesh[1].ravel()), k_fold, 2)

    testColormap = ListedColormap(['#FF7373', '#00CC00'])
    classColormap = ListedColormap(['#FF0000', '#67E667'])

    plt.pcolormesh(testMesh[0],
                  testMesh[1],
                  np.asarray(testMeshLabels).reshape(testMesh[0].shape),
                  cmap=testColormap)

    plt.scatter([trainDots[i][0][0] for i in range(len(trainDots))],
               [trainDots[i][0][1] for i in range(len(trainDots))],
               c=[trainDots[i][1] for i in range(len(trainDots))],
               cmap=classColormap)

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()
