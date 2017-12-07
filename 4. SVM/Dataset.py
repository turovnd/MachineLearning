import numpy as np
import random


class Dataset:
    def __init__(self, filename, trainDots=50):
        self.filename = filename
        self.data = []
        self.train = []
        self.num_train = trainDots
        self.test = []
        file = open(filename)

        for line in file:
            dot_x, dot_y, dot_class = line.split(',')
            if int(dot_class) == 0:
                dot_class = -1
            self.data.append([[float(dot_x), float(dot_y)], float(dot_class)])

        file.close()

        if self.num_train + 5 > len(self.data):
            print("error: number of train dots more than number of all dots in dataset")
            exit(1)

    ##
    # Reset train and test Dots
    ##
    def reset(self):
        random.shuffle(self.data)
        self.train = self.data[0:self.num_train]
        self.test = self.data[self.num_train + 1:len(self.data)]

    ##
    # Get Dots By Mode
    # @mode = train|test
    ##
    def getDotsByMode(self, mode, as_npArr):
        if as_npArr:
            if mode == "train":
                return np.array([self.train[i][0] for i in range(len(self.train))]), np.array([self.train[i][1] for i in range(len(self.train))])
            if mode == "test":
                return np.array([self.test[i][0] for i in range(len(self.test))]), np.array([self.test[i][1] for i in range(len(self.test))])
        else:
            if mode == "train":
                return [self.train[i][0] for i in range(len(self.train))], [self.train[i][1] for i in range(len(self.train))]
            if mode == "test":
                return [self.test[i][0] for i in range(len(self.test))], [self.test[i][1] for i in range(len(self.test))]
