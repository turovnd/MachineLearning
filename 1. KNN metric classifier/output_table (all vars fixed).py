from DatasetProcessing import *
from Plot import *
from Statistic import *
from tabulate import tabulate


def make_computing(trainDots, testDots, k_neighbors, kernelFunction, metrics, coordinateTransformation):
    testClasses = [testDots[i][1] for i in range(len(testDots))]
    testClassesClassified = DatasetProcessing.classifyKNNCircle(trainDots, testDots, k_neighbors, kernelFunction,
                                                                metrics, coordinateTransformation)

    F1_measure = Statistic.computingF1_measure(testClassesClassified, testClasses)
    Recall = Statistic.computingRecall(testClassesClassified, testClasses)
    Specificity = Statistic.computingSpecificity(testClassesClassified, testClasses)
    Precision = Statistic.computingPrecision(testClassesClassified, testClasses)
    Accuracy = Statistic.computingAccuracy(testClassesClassified, testClasses)

    return [F1_measure, Recall, Specificity, Precision, Accuracy]


def get_table_row(trainDots, testDots, k_fold, k_neighbors, kernelFunction, metrics, coordinateTransformation):
    F1_measure = []
    Recall = []
    Specificity = []
    Precision = []
    Accuracy = []

    for i in range(k_fold):
        arr = make_computing(trainDots[i], testDots[i], k_neighbors, kernelFunction, metrics, coordinateTransformation)
        F1_measure.append(arr[0])
        Recall.append(arr[1])
        Specificity.append(arr[2])
        Precision.append(arr[3])
        Accuracy.append(arr[4])

    F1_measure = sum(F1_measure[i] for i in range(len(F1_measure))) / len(F1_measure)
    Recall = sum(Recall[i] for i in range(len(Recall))) / len(Recall)
    Specificity = sum(Specificity[i] for i in range(len(Specificity))) / len(Specificity)
    Precision = sum(Precision[i] for i in range(len(Precision))) / len(Precision)
    Accuracy = sum(Accuracy[i] for i in range(len(Accuracy))) / len(Accuracy)

    return [kernelFunction, metrics, coordinateTransformation, F1_measure, Recall, Specificity, Precision, Accuracy]


if __name__ == '__main__':
    # ядро используемой функции
    kernelFunctions = ["none", "gaussian", "logistic"]

    # метрика расстояния
    metrics = ["manhattan", "euclidean"]

    # пространственное преобразование
    coordinateTransformations = ["none", "elliptic", "hyperbolic"]

    # количетсво тренировочных точек
    number_trainDots = 20

    # количество соседий
    k_neighbors = 10

    # k-fold - количество итераций, после которых значения апроксимируются
    k_fold = 2

    # Входной датасет, не изменяется в пределах одного `coordinateTransformation`
    # индекс датасето соответстует индексу `coordinateTransformations`
    data = [
        DatasetProcessing.getDataset('dataset.txt', coordinateTransformations[0]),
        DatasetProcessing.getDataset('dataset.txt', coordinateTransformations[1]),
        DatasetProcessing.getDataset('dataset.txt', coordinateTransformations[2])
    ]

    print("Data length   = ", len(data[0]))
    print("Training dots = ", number_trainDots)
    print("Test dots     = ", len(data[0]) - number_trainDots)
    print("k_neighbors   = ", k_neighbors)
    print("k_fold        = ", k_fold)

    trainDots = [[[] for i in range(k_fold)] for i in range(len(coordinateTransformations))]
    testDots = [[[] for i in range(k_fold)] for i in range(len(coordinateTransformations))]

    for i in range(len(coordinateTransformations)):
        for j in range(k_fold):
            trainD, testD = DatasetProcessing.getTrainTestDots(number_trainDots, data[i])
            trainDots[i][j] = trainD
            testDots[i][j] = testD

    rows = []

    for coord in range(len(coordinateTransformations)):
        for metric in range(len(metrics)):
            for fun in range(len(kernelFunctions)):
                rows.append(get_table_row(trainDots[coord],
                                          testDots[coord],
                                          k_fold, k_neighbors,
                                          kernelFunctions[fun], metrics[metric],
                                          coordinateTransformations[coord]))

        rows.append([])

    print('_' * 162, '\n')

    print(tabulate(rows,
                   headers=["Kernel functions",
                            "Metrics for configuring kNN", "Spatial coordinate transformations", "F1-measure",
                            "Recall", "Specificity", "Precision", "Accuracy"], tablefmt='orgtbl'))
