from DatasetProcessing import *
from Visualization import *
from GradientDescent import *

if __name__ == '__main__':
    data = DatasetProcessing.getDataset('dataset.txt')
    normalizeData = DatasetProcessing.getNormalizeDataset(data)

    print("Data =", data)
    print("normalizeData =", normalizeData)
    print("Rooms =", len(data))

    #TODO в одном ненормализованный и нормализованный
    #Visualization.buildStartDatasetPlot(normalizeData)
    print(GradientDescent.calculateNewPrice(data, 0.05, 2000))