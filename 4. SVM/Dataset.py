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
            self.data.append([[float(dot_x), float(dot_y)], int(dot_class)])

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
    # Get Dots By Class
    # @mode        = train|test
    # @mode_class  = 1|0
    ##
    def getDotsByClass(self, mode, mode_class):
        dots = []
        if mode == "train":
            for i in range(len(self.train)):
                if self.train[i][1] == mode_class:
                    dots.append(self.train[i][0])
        if mode == "test":
            for i in range(len(self.test)):
                if self.test[i][1] == mode_class:
                    dots.append(self.test[i][0])
        return dots
