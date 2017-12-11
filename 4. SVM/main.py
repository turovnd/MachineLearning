from tabulate import tabulate
import scipy.stats as stats
from Dataset import Dataset
from CrossValidation import Validator
from kernel import Kernel
from metric import Metric

if __name__ == "__main__":
    ##
    # kernel        => 'gaussian' || 'polynomial' || 'linear'
    # metric        => 'euclidean' || 'manhattan'
    # svm_C         => Float || None
    # k_neighbors   => Number
    # show_plot     => False || True
    ##
    kernel = Kernel.get('polynomial')
    metric = Metric.get('manhattan')
    svm_C = 1000
    show_plot = False
    k_neighbors = 5

    data = Dataset('input/chips.txt')
    validator = Validator()

    print("SVM")
    y_predict_SVM, f_measure, matrix = validator.svm_validate(data, kernel, svm_C, show_plot)
    table = [["T", matrix[0][0], matrix[0][1]], ["F", matrix[1][0], matrix[1][1]]]
    print(tabulate(table, headers=["", "P", "N"], tablefmt='orgtbl'))
    print("F-measure: " + str(f_measure) + "\n")

    print("kNN")
    y_predict_kNN, f_measure, matrix = validator.knn_validate(data, kernel, metric, k_neighbors, show_plot)
    table = [["T", matrix[0][0], matrix[0][1]], ["F", matrix[1][0], matrix[1][1]]]
    print(tabulate(table, headers=["", "P", "N"], tablefmt='orgtbl'))
    print("F-measure: " + str(f_measure) + "\n")

    z_statistic, p_value = stats.wilcoxon(y_predict_SVM, y_predict_kNN)
    print("P-value: " + str(p_value) + "\n")
