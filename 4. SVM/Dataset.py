import numpy as np
import random


class Dataset:
    def __len__(self):
        return len(self.data)

    def __init__(self, filename):
        self.filename = filename
        self.data = []
        self.train = []
        self.test = []

        ff = open(filename)

        for line in ff:
            dot_x, dot_y, dot_class = line.split(',')
            if int(dot_class) == 0:
                dot_class = -1
            self.data.append([[float(dot_x), float(dot_y)], float(dot_class)])

        ff.close()
        random.shuffle(self.data)

    def reset(self):
        random.shuffle(self.data)

    def updateTrainTest(self, ind):
        self.train = self.data[0:ind] + self.data[ind+1:len(self.data)]
        self.test = self.data[ind]

    ##
    # Get Dots By Mode
    # @mode = train|test
    ##
    def getDotsByMode(self, mode, as_npArr):
        if as_npArr:
            if mode == "train":
                return np.array([self.train[i][0] for i in range(len(self.train))]), \
                       np.array([self.train[i][1] for i in range(len(self.train))])
            if mode == "test":
                return np.array([self.test[0]]), np.array([self.test[1]])
        else:
            if mode == "train":
                return [self.train[i][0] for i in range(len(self.train))], \
                       [self.train[i][1] for i in range(len(self.train))]
            if mode == "test":
                return [self.test[0]], [self.test[1]]
