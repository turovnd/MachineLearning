import numpy as np
from kernel import Kernel

class KNN(object):
    def __init__(self, kernel, neighbors=5):
        self.kernel = Kernel.get(kernel)
        self.neighbors = neighbors

    def fit(self, dots, cls):
        self.trainDots = dots
        self.trainClass = cls

    def classify(self, dot):
        testDist = []
        for i in range(len(self.trainDots)):
            testDist.append([np.math.sqrt(np.math.pow(self.trainDots[i][0] - dot[0], 2) + np.math.pow(self.trainDots[i][1] - dot[1], 2)),
                             self.trainClass[i], dot, self.trainDots[i]])

        n = 1
        while n < len(testDist):
            for i in range(len(testDist) - n):
                if testDist[i][0] > testDist[i + 1][0]:
                    testDist[i], testDist[i + 1] = testDist[i + 1], testDist[i]
            n += 1

        class_0 = 0
        class_1 = 0

        for i in range(self.neighbors):
            if testDist[i][1] == 1:
                class_1 += self.kernel(np.array(testDist[i][2]), np.array(testDist[i][3]))
            else:
                class_0 += self.kernel(np.array(testDist[i][2]), np.array(testDist[i][3]))

        if class_0 > class_1:
            return -1
        else:
            return 1

    def predict(self, X):
        classes = []
        for i in range(len(X)):
            cls = self.classify(X[i])
            self.trainDots.append(X[i])
            self.trainClass.append(cls)
            classes.append(cls)
        return np.array(classes)
