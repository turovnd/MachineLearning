from DatasetProcessing import *
from Visualization import *
from GradientDescent import *

if __name__ == '__main__':
    data = DatasetProcessing.getDataset('dataset.txt')
    normalizeData = DatasetProcessing.getNormalizeDataset(data)

    print("Data =", data)
    print("normalizeData =", normalizeData)
    print("Rooms =", len(data))
    area, rooms, price = DatasetProcessing.getSeparetedData(normalizeData)
    print("area =", area)
    print("rooms =", rooms)
    print("price =", price)
    #TODO в одном ненормализованный и нормализованный
    #Visualization.buildStartDatasetPlot(normalizeData)
    (j_hist, w, newPrice) = GradientDescent.calculateNewPrice(data, 0.99, 2650, 0.000001)

    print("Nprice =", newPrice)
    print("j_hist =", j_hist)

    Visualization.buildNewDatasetPlot(normalizeData, newPrice)
    # Visualization.buildErrorDatasetPlot(j_hist)
