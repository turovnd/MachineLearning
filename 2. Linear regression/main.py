from DatasetProcessing import *
from Visualization import *

if __name__ == '__main__':
    data = DatasetProcessing.getDataset('dataset.txt')
    print("Лист =", data)
    # for i in range(len(data)):
    #     print("area =", data[i][0])
    #     print("rooms =", data[i][1])
    #     print("price =", data[i][2])

    print("Количество комнат =", len(data))
    Visualization.buildStartDatasetPlot(data)
