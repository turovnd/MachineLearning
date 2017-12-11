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
    metric = Metric.get('euclidean')
    svm_C = 1
    show_plot = False
    k_neighbors = 5

    data = Dataset('input/chips.txt')
    validator = Validator()

    print("SVM")
    f_measure, matrix = validator.svm_validate(data, kernel, svm_C, show_plot)
    print(f_measure)
    print(matrix)

    print("kNN")
    f_measure, matrix = validator.knn_validate(data, kernel, metric, k_neighbors, show_plot)
    print(f_measure)
    print(matrix)
