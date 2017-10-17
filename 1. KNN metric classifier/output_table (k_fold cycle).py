from DatasetProcessing import *
from Plot import *
from Statistic import *
from tabulate import tabulate
import matplotlib.pyplot as plt


def make_computing(trainDots, testDots, k_neighbors, kernelFunction, metrics, coordinateTransformation):
    testClasses = [testDots[i][1] for i in range(len(testDots))]
    testClassesClassified = DatasetProcessing.classifyKNNCircle(trainDots, testDots, k_neighbors, kernelFunction,
                                                                metrics, coordinateTransformation)

    return  Statistic.computingF1_measure(testClassesClassified, testClasses)


def get_table_row(trainDots, testDots, k_fold, k_neighbors, kernelFunction, metrics, coordinateTransformation):
    F1_measure = []

    for i in range(k_fold):
        F1_measure.append(
            make_computing(trainDots[i], testDots[i], k_neighbors, kernelFunction, metrics, coordinateTransformation)
        )

    F1_measure = sum(F1_measure[i] for i in range(len(F1_measure))) / len(F1_measure)

    return [k_fold, kernelFunction, metrics, coordinateTransformation, F1_measure]


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
    k_fold = [i for i in range(1,11)]

    # Входной датасет, не изменяется в пределах одного `coordinateTransformation`
    # индекс датасето соответстует индексу `coordinateTransformations`
    data = [
        DatasetProcessing.getDataset('dataset.txt', coordinateTransformations[0]),
        DatasetProcessing.getDataset('dataset.txt', coordinateTransformations[1]),
        DatasetProcessing.getDataset('dataset.txt', coordinateTransformations[2])
    ]

    trainDots = [[[] for i in range(max(k_fold))] for i in range(len(coordinateTransformations))]
    testDots = [[[] for i in range(max(k_fold))] for i in range(len(coordinateTransformations))]

    for i in range(len(coordinateTransformations)):
        for j in range(max(k_fold)):
            trainD, testD = DatasetProcessing.getTrainTestDots(number_trainDots, data[i])
            trainDots[i][j] = trainD
            testDots[i][j] = testD

    rows = []

    coord_ind = 2 # 0|1|2
    coord  = coordinateTransformations[coord_ind]
    metric = metrics[1]
    kFun   = kernelFunctions[2]

    for k_f in k_fold:
        row = get_table_row(trainDots[coord_ind],testDots[coord_ind],k_f, k_neighbors, kFun, metric, coord)
        rows.append(row)
        print(row)

    print('_' * 162, '\n')


    print("Data length   = ", len(data[0]))
    print("Training dots = ", number_trainDots)
    print("Test dots     = ", len(data[0]) - number_trainDots)
    print("k_neighbors   = ", k_neighbors)

    print('_' * 162, '\n')

    print(tabulate(rows,
                   headers=["k_fold","Kernel functions",
                            "Metrics for configuring kNN", "Spatial coordinate transformations", "F1-measure"], tablefmt='orgtbl'))


    points = plt.plot([rows[i][0] for i in range(len(rows))],
                         [rows[i][4] for i in range(len(rows))], 'go')

    # подпись осей
    plt.xlabel('$k-fold$')
    plt.ylabel('$F1-measure$')
    plt.grid(True)
    plt.show()
