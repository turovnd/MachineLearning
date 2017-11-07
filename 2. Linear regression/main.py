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

    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 10, 50000, 0.00000001)
    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 1, 50000, 0.00000001)
    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 0.1, 50000, 0.00000001)
    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 0.01, 50000, 0.00000001)
    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 0.05, 50000, 0.00000001)
    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 0.001, 50000, 0.00000001)
    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 0.005, 50000, 0.00000001)
    (lastIteration, MSE, w, newPrice, w1, w2) = GradientDescent.calculateGradientDescent(data, 0.0001, 2400, 0.0000001)
    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 0.0001, 24000, 0.00000001)
    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 0.0005, 50000, 0.00000001)
    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 0.00001, 50000, 0.00000001)
    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 0.00005, 50000, 0.00000001)
    # (j_hist, w, newPrice, w0, w1) = GradientDescent.calculateGradientDescent(data, 0.000001, 50000, 0.00000001)

    # print("j_hist =", j_hist)
    # newPrice_NP = np.asarray(newPrice)
    # price_NP = np.asarray(price)
    # error = newPrice_NP - price_NP
    # print(error)
    Visualization.build3DStartDataset(data)
    Visualization.build3DRegressionLinear(normalizeData, w1, w2)
    Visualization.build2DInfo(price, newPrice, MSE, lastIteration)
