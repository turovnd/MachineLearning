from DatasetProcessing import *
from Plot import *
from Statistic import *
from tabulate import tabulate


def make_computing(data, number_trainDots, k_neighbors, kernelFunction, metrics, coordinateTransformation):
    trainDots, testDots = DatasetProcessing.getTrainTestDots(number_trainDots, data)
    testClasses = [testDots[i][1] for i in range(len(testDots))]
    testClassesClassified = DatasetProcessing.classifyKNNCircle(trainDots, testDots, k_neighbors, kernelFunction,
                                                                metrics, coordinateTransformation)

    F1_measure = Statistic.computingF1_measure(testClassesClassified, testClasses)
    Recall = Statistic.computingRecall(testClassesClassified, testClasses)
    Specificity = Statistic.computingSpecificity(testClassesClassified, testClasses)
    Precision = Statistic.computingPrecision(testClassesClassified, testClasses)
    Accuracy = Statistic.computingAccuracy(testClassesClassified, testClasses)

    return [F1_measure, Recall, Specificity, Precision, Accuracy]


if __name__ == '__main__':
    # kernelFunction: ядро используемой функции [none||gaussian||logistic]
    # metrics: метрика расстояния [manhattan||euclidean]
    # coordinateTransformation - пространственное преобразование [none||elliptic||hyperbolic]

    kernelFunction = "gaussian"
    metrics = "euclidean"
    coordinateTransformation = "elliptic"
    number_trainDots = 50
    k_neighbors = 15
    k = 10

    data = DatasetProcessing.getDataset('dataset.txt', coordinateTransformation)

    F1_measure = []
    Recall = []
    Specificity = []
    Precision = []
    Accuracy = []

    for i in range(k):
        arr = make_computing(data, number_trainDots, k_neighbors, kernelFunction, metrics, coordinateTransformation)
        F1_measure.append(arr[0])
        Recall.append(arr[1])
        Specificity.append(arr[2])
        Precision.append(arr[3])
        Accuracy.append(arr[4])

    F1_measure = sum(F1_measure[i] for i in range(len(F1_measure)))/len(F1_measure)
    Recall = sum(Recall[i] for i in range(len(Recall)))/len(Recall)
    Specificity = sum(Specificity[i] for i in range(len(Specificity)))/len(Specificity)
    Precision = sum(Precision[i] for i in range(len(Precision)))/len(Precision)
    Accuracy = sum(Accuracy[i] for i in range(len(Accuracy)))/len(Accuracy)

    print(tabulate([[number_trainDots, len(data) - number_trainDots, k_neighbors, k, kernelFunction,
                    metrics, coordinateTransformation, F1_measure, Recall, Specificity, Precision, Accuracy]],
                    headers=["Training dots", "Test dots", "k (neighbors)", "k", "Kernel functions",
                                "Metrics for configuring kNN", "Spatial coordinate transformations", "F1-measure",
                                "Recall", "Specificity", "Precision", "Accuracy"], tablefmt='orgtbl'))
