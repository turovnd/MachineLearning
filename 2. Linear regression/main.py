from DatasetProcessing import *
from Visualization import *

if __name__ == '__main__':
    data = DatasetProcessing.getDataset('dataset.txt')
    print("Лист =", data)

    print("Количество комнат =", len(data))
    normalizeData = DatasetProcessing.getNormalizeDataset(data)
    #TODO в одном ненормализованный и нормализованный
    Visualization.buildStartDatasetPlot(normalizeData)
