from Dataset import Dataset
from CrossValidation import Validator
from svm import SVM
from knn import KNN


if __name__ == "__main__":
    ##
    # kernel        => 'gaussian' || 'polynomial' || 'linear'
    # svm_C         => Float || None
    # k_neighbors   => Number
    # cross_fold    => Number
    # show_plot     => False || True
    ##
    kernel = 'polynomial'
    svm_C = 1
    cross_fold = 1
    show_plot = False
    k_neighbors = 5

    data = Dataset('input/chips.txt', trainDots=80)
    validator = Validator()

    print("SVM")
    f_measure, matrix = validator.svm_validate(data, kernel, svm_C, cross_fold, show_plot)
    print(f_measure)
    print (matrix)

    print("kNN")
    f_measure, matrix = validator.knn_validate(data, kernel, k_neighbors, cross_fold, show_plot)
    print(f_measure)
    print (matrix)
