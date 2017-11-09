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
    print("-----------------------------------------------------------------------------------------------")
    #TODO подбор параметров
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.01)
    # # print("MSE =", MSE)  # 0.35
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.001)
    # # print("MSE =", MSE)  # 0.35
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0001)
    # # print("MSE =", MSE)  # 0.35
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.00001)
    # # print("MSE =", MSE)  # 0.58
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.000001)
    # # print("MSE =", MSE)  # 1.47
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000001)
    # # print("MSE =", MSE)  # 0.9
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.00000001)
    # # print("MSE =", MSE)  # 1.66
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.000000001)
    # print("MSE =", MSE)  # 1.57
    (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
        GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)
    # # print("MSE =", MSE)  # 1.68
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 24000, 0.00000000001)
    # # print("MSE =", MSE)  # 1.68
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.000000000001)
    # # print("MSE =", MSE)  # 1.67
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000000001)
    # print("MSE =", MSE)  # 1.68
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)
    # # print("MSE =", MSE)  # 1.68
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)
    # (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
    #     GradientDescent.calculateGradientDescent(data, 0.4459, 50000, 0.0000000001)


    # print("MSE =", MSE)  # --

    # print("MSE =", MSE[1000], MSE[2000], MSE[3000], MSE[4000], MSE[5000], MSE[10000])
    # newPrice_NP = np.asarray(newPrice)
    # price_NP = np.asarray(price)
    # error = newPrice_NP - price_NP
    # print(error)
    # print(weight_hist1)
    # print(weight_hist2)
    # print("-----------------------------------------------------------------------------------------------")
    Visualization.build3DStartDataset(data)
    Visualization.build3DCostFunction(weight_hist1, weight_hist2, MSE, lastIteration)
    Visualization.build3DRegressionLinear(normalizeData, wLast, newPrice)
    Visualization.build2DInfo(price, newPrice, MSE, lastIteration)
