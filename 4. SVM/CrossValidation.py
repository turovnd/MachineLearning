import numpy as np
from svm import SVM
from knn import KNN
from Plot import Plot


##
# Get F-measure by Confusion Matrix
# @matrix => [[Number, Number], [Number, Number]]
##
def get_f_measure(matrix):
    if matrix[0][0] != 0:
        recall = float(matrix[0][0]) / (float(matrix[0][0]) + float(matrix[1][1]))
        precision = float(matrix[0][0]) / (float(matrix[0][0]) + float(matrix[1][0]))
        return 2 * precision * recall / (precision + recall)
    else:
        return 0


##
# Get Metrics By Array
# @predict  => [Number, Number]
# @start    => [Number, Number]
##
def get_metrics(predict, start):
    # true positive is a spam which is predicted well
    # true negative is a not spam which is predicted well
    # false positive is a spam which is not predicted well
    # false negative is a not spam which is not predicted well
    # [[ true positive, true negative ], [ false positive, false negative ]]
    matrix = [[0, 0], [0, 0]]
    for i in range(len(predict)):
        if predict[i] == start[i]:
            if predict[i] == 1.0:
                matrix[0][0] += 1
            else:
                matrix[0][1] += 1
        else:
            if predict[i] == 1.0:
                matrix[1][0] += 1
            else:
                matrix[1][1] += 1

    return matrix


class Validator(object):
    def __init__(self):
        pass

    @staticmethod
    def svm_validate(data, kernel, svm_C, show_plot):
        plot = Plot()
        matrix_full = [[0, 0], [0, 0]]
        y_predict_arr = []

        for i in range(len(data)):
            data.updateTrainTest(i)
            trainDots, trainClass = data.getDotsByMode('train', True)
            testDots, testClass = data.getDotsByMode('test', True)

            clf = SVM(kernel=kernel, C=svm_C)
            clf.fit(trainDots, trainClass)
            y_predict = clf.predict(testDots)
            y_predict_arr.append(y_predict[0])

            if show_plot:
                plot.smv(trainDots[trainClass == 1], trainDots[trainClass == -1], clf, testDots[0], y_predict[0])

            matrix = get_metrics(y_predict, testClass)
            matrix_full[0][0] += matrix[0][0]
            matrix_full[0][1] += matrix[0][1]
            matrix_full[1][0] += matrix[1][0]
            matrix_full[1][1] += matrix[1][1]

        return y_predict_arr, get_f_measure(matrix_full), matrix_full

    @staticmethod
    def knn_validate(data, kernel, metric, k_neighbors, show_plot):
        plot = Plot()
        matrix_full = [[0, 0], [0, 0]]
        y_predict_arr = []
        for i in range(len(data)):
            data.updateTrainTest(i)
            trainDots, trainClass = data.getDotsByMode('train', False)
            testDots, testClass = data.getDotsByMode('test', False)

            knn = KNN(kernel=kernel, metric=metric, neighbors=k_neighbors)
            knn.fit(trainDots, trainClass)
            y_predict, distance = knn.predict(testDots)
            y_predict_arr.append(y_predict[0])

            if show_plot:
                tDots = np.array(trainDots)
                tCls = np.array(trainClass)
                plot.knn(tDots[tCls == 1.0], tDots[tCls == -1.0], distance, testDots[0], y_predict[0])

            matrix = get_metrics(y_predict, testClass)
            matrix_full[0][0] += matrix[0][0]
            matrix_full[0][1] += matrix[0][1]
            matrix_full[1][0] += matrix[1][0]
            matrix_full[1][1] += matrix[1][1]

        return y_predict_arr, get_f_measure(matrix_full), matrix_full
