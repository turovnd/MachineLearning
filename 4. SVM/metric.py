from numpy import math


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def euclidean(x1, x2):
        return math.sqrt(math.pow(x1[0] - x2[0], 2) + math.pow(x1[1] - x2[1], 2))

    @staticmethod
    def manhattan(x1, x2):
        return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

    @staticmethod
    def get(metric):
        if metric == 'euclidean':
            return Metric.euclidean
        elif metric == 'manhattan':
            return Metric.manhattan
