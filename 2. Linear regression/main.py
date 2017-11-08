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

    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 20, 50000, 0.00000001)
    # print("MSE =", MSE)  # 0.35
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 10, 50000, 0.00000001)
    # print("MSE =", MSE)  # 0.35
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 1, 50000, 0.00000001)
    # print("MSE =", MSE)  # 0.35
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 0.1, 50000, 0.00000001)
    # print("MSE =", MSE)  # 0.58
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 0.01, 50000, 0.00000001)
    # print("MSE =", MSE)  # 1.47
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 0.05, 50000, 0.00000001)
    # print("MSE =", MSE)  # 0.9
    (lastIteration, MSE, w, newPrice) = \
        GradientDescent.calculateGradientDescent(data, 0.001, 50000, 0.00000001)
    # print("MSE =", MSE)  # 1.66
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 0.005, 50000, 0.00000001)
    # print("MSE =", MSE)  # 1.57
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 0.0001, 50000, 0.00000001)
    # print("MSE =", MSE)  # 1.68
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 0.0001, 24000, 0.00000001)
    # print("MSE =", MSE)  # 1.68
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 0.0005, 50000, 0.00000001)
    # print("MSE =", MSE)  # 1.67
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 0.00001, 50000, 0.00000001)
    # print("MSE =", MSE)  # 1.68
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 0.00005, 50000, 0.00000001)
    # print("MSE =", MSE)  # 1.68
    # (lastIteration, MSE, w, newPrice) = \
    #     GradientDescent.calculateGradientDescent(data, 0.000001, 50000, 0.00000001)
    # print("MSE =", MSE)  # --

    # print("MSE =", MSE[1000], MSE[2000], MSE[3000], MSE[4000], MSE[5000], MSE[10000])
    # newPrice_NP = np.asarray(newPrice)
    # price_NP = np.asarray(price)
    # error = newPrice_NP - price_NP
    # print(error)

    Visualization.build3DStartDataset(data)
    Visualization.build3DRegressionLinear(normalizeData, w, newPrice)
    Visualization.build2DInfo(price, newPrice, MSE, lastIteration)
