import numpy as np


class KNN(object):
    def __init__(self, kernel, metric, neighbors=5):
        self.kernel = kernel
        self.metric = metric
        self.neighbors = neighbors

    def fit(self, dots, cls):
        self.trainDots = dots
        self.trainClass = cls

    def classify(self, dot):
        distances = []
        for i in range(len(self.trainDots)):
            distances.append([self.metric(self.trainDots[i], dot), self.trainClass[i], self.trainDots[i]])

        closest_distances = sorted(distances)[:self.neighbors]

        class_0 = 0.0
        class_1 = 0.0

        for distance in closest_distances:
            if distance[1] == 1.0:
                class_1 += self.kernel(np.array(dot), np.array(distance[2]))
            else:
                class_0 += self.kernel(np.array(dot), np.array(distance[2]))

        if class_0 > class_1:
            return [-1.0], closest_distances
        else:
            return [1.0], closest_distances

    def predict(self, dot):
        return self.classify(dot[0])
