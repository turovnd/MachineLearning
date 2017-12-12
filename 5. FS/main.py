import numpy as np
from numpy import genfromtxt
from tabulate import tabulate
from Statistic import Statistic
from Kernel import Kernel
from Metric import Metric
from svm import SVM


def build(kernel, metric, keys_limit, svm_C, logs):
    trainX = genfromtxt('input/arcene_train.data', delimiter=' ')
    trainY = genfromtxt('input/arcene_train.labels', delimiter=' ')
    validX = genfromtxt('input/arcene_valid.data', delimiter=' ')
    validY = genfromtxt('input/arcene_valid.labels', delimiter=' ')

    keys = metric.build(trainX.transpose(), trainY, logs=logs, limit=keys_limit)

    tX = []
    for x in trainX:
        tX.append(np.take(x, keys))

    tX = np.array(tX)

    clf = SVM(kernel=kernel.kernel, C=svm_C)
    clf.fit(tX, trainY)

    vX = []
    for x in validX:
        vX.append(np.take(x, keys))

    vX = np.array(vX)

    predict_arr = [clf.predict(x) for x in vX]
    confusion_matrix = Statistic.get_metrics(predict_arr, validY)
    f_measure = Statistic.get_f_measure(confusion_matrix)

    return predict_arr, confusion_matrix, f_measure


def create_table(kernels, metrics, keys_limit, svm_C):
    output_arr = []
    table = []
    for kernel in kernels:
        for metric in metrics:
            arr, confusion_matrix, f_measure = build(kernel, metric, keys_limit, svm_C, False)
            output_arr.append([metric.name, arr])
            table.append([kernel.name, metric.name, keys_limit, svm_C, f_measure, confusion_matrix])

    print("\nComparable Table")
    print(tabulate(table,
                   headers=["Kernel", "Metric", "Filter limit", "SVM C", "F-mature", "Confusion Matrix"],
                   tablefmt='orgtbl'))

    return output_arr

##
# kernel        => 'gaussian' || 'polynomial' || 'linear'
# metric        => 'pearson' || 'spearman' || 'ig'
# svm_C         => Float || None
# keys_limit    => Number
# show_plot     => False || True
# logs          => False || True
##
if __name__ == "__main__":
    kernel = Kernel('linear')
    metric = Metric('ig')
    keys_limit = 1000
    svm_C = 1
    show_plot = False
    logs = False

    arrays = create_table([Kernel('linear')], [Metric('pearson'), Metric('spearman'), Metric('ig')], keys_limit, svm_C)



