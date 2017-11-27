import numpy as np
from CustomSVM import CustomSVM
from Dataset import Dataset
from sklearn import metrics
from sklearn.svm import SVC
if __name__ == "__main__":

    trainDots = 50

    data = Dataset('input/chips.txt', trainDots)
    data.reset()

    svm = CustomSVM()
    svm.fit(data={
        0: np.array(data.getDotsByClass('train', 0)),
        1: np.array(data.getDotsByClass('train', 1))
    })


    # predict_us = [[0, 10],
    #               [1, 3],
    #               [3, 4],
    #               [3, 5],
    #               [5, 5],
    #               [5, 6],
    #               [6, -5],
    #               [5, 8]]
    #
    # for p in predict_us:
    #     svm.predict(p)

    svm.visualize()
