import numpy as np
from svm import SVM
from knn import KNN
from Plot import Plot


def get_metrics(predict, start):
    # true positive is a spam which is predicted well
    # true negative is a not spam which is predicted well
    # false positive is a spam which is not predicted well
    # false negative is a not spam which is not predicted well
    # [[ true positive, true negative ], [ false positive, false negative ]]
    matrix = [[0, 0], [0, 0]]
    for i in range(len(predict)):
        if predict[i] == start[i]:
            if int(predict[i]) == 1:
                matrix[0][0] += 1
            else:
                matrix[1][0] += 1
        else:
            if int(predict[i]) == 1:
                matrix[0][1] += 1
            else:
                matrix[1][1] += 1

    if matrix[0][0] != 0:
        recall = float(matrix[0][0]) / (float(matrix[0][0]) + float(matrix[1][1]))
        precision = float(matrix[0][0]) / (float(matrix[0][0]) + float(matrix[1][0]))
        f_measure = 2 * precision * recall / (precision + recall)
    else:
        f_measure = 0

    return f_measure, matrix



class Validator(object):
    def __init__(self):
        pass

    @staticmethod
    def svm_validate(data, kernel, svm_C, cross_fold, show_plot):
        plot = Plot()
        matrix_full = [[0, 0], [0, 0]]
        f_measures = []

        for i in range(cross_fold):
            data.reset()
            trainDots, trainClass = data.getDotsByMode('train', True)
            testDots, testClass = data.getDotsByMode('test', True)

            clf = SVM(kernel=kernel, C=svm_C)
            clf.fit(trainDots, trainClass)
            y_predict = clf.predict(testDots)

            if show_plot:
                plot.smv(trainDots[trainClass == 1], trainDots[trainClass == -1], clf)

            f_measure, matrix = get_metrics(y_predict, testClass)
            matrix_full[0][0] += matrix[0][0]
            matrix_full[0][1] += matrix[0][1]
            matrix_full[1][0] += matrix[1][0]
            matrix_full[1][1] += matrix[1][1]
            f_measures.append(f_measure)

        return sum(f_measures) / len(f_measures), matrix_full

    @staticmethod
    def knn_validate(data, kernel, k_neighbors, cross_fold, show_plot):
        plot = Plot()
        matrix_full = [[0, 0], [0, 0]]
        f_measures = []

        for i in range(cross_fold):
            data.reset()
            trainDots, trainClass = data.getDotsByMode('train', False)
            testDots, testClass = data.getDotsByMode('test', False)

            knn = KNN(kernel=kernel, neighbors=k_neighbors)
            knn.fit(trainDots, trainClass)
            y_predict = knn.predict(testDots)

            # if show_plot:
            # plot.smv(trainDots[trainClass == 1], trainDots[trainClass == -1], clf)

            f_measure, matrix = get_metrics(y_predict, testClass)
            matrix_full[0][0] += matrix[0][0]
            matrix_full[0][1] += matrix[0][1]
            matrix_full[1][0] += matrix[1][0]
            matrix_full[1][1] += matrix[1][1]
            f_measures.append(f_measure)

        return sum(f_measures) / len(f_measures), matrix_full
