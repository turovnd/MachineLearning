import numpy as np
from numpy import genfromtxt
from tabulate import tabulate
from Statistic import Statistic
from Kernel import Kernel
from Metric import Metric
from svm import SVM

if __name__ == "__main__":
    ##
    # kernel        => 'gaussian' || 'polynomial' || 'linear'
    # metric        => 'pearson' || 'spearman' || 'ig'
    # svm_C         => Float || None
    # keys_limit    => Number
    # show_plot     => False || True
    # logs          => False || True
    ##
    kernel = Kernel.get('linear')
    metric = Metric('ig')
    keys_limit = 1000
    svm_C = 1
    show_plot = False
    logs = False

    trainX = genfromtxt('input/arcene_train.data', delimiter=' ')
    trainY = genfromtxt('input/arcene_train.labels', delimiter=' ')
    validX = genfromtxt('input/arcene_valid.data', delimiter=' ')
    validY = genfromtxt('input/arcene_valid.labels', delimiter=' ')

    keys = metric.build(trainX.transpose(), trainY, logs=logs, limit=keys_limit)

    X = []
    for x in trainX:
        X.append(np.take(x, keys))

    X = np.array(X)
    Y = trainY

    clf = SVM(kernel=kernel, C=svm_C)
    clf.fit(X, Y)

    X = []
    for x in validX:
        X.append(np.take(x, keys))

    X = np.array(X)
    Y = validY

    predict_arr = [clf.predict(x) for x in X]
    confusion_matrix = Statistic.get_metrics(predict_arr, validY)
    f_measure = Statistic.get_f_measure(confusion_matrix)

    print("\nF-measure: " + str(f_measure) + "\n")
    print(tabulate(
        [["T", confusion_matrix[0][0], confusion_matrix[0][1]], ["F", confusion_matrix[1][0], confusion_matrix[1][1]]],
        headers=["", "P", "N"], tablefmt='orgtbl'))
