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

    areaInputList = []
    roomsInputList = []
    print("write two different values")
    while True:
        try:
            areaInputVar = input("write area ('q' to exit): ")
            if areaInputVar == 'q':
                break
            else:
                areaInputList.append(float(areaInputVar))

            roomsInputVar = input("write number of rooms ('q' to exit): ")
            if roomsInputVar == 'q':
                break
            else:
                roomsInputList.append(float(roomsInputVar))
        except ValueError:
            print("Wrong format\n")
            continue

    Visualization.build3DStartDataset(data)
    Visualization.build3DCostFunction(weight_hist1, weight_hist2, MSE, lastIteration)
    if (len(areaInputList) != 0):
        priceNormalizeInputList = GradientDescent.calculateInputPrice(areaInputList, roomsInputList, wLast)
        normalizeDataInput = DatasetProcessing.getCombinedInputData(areaInputList, roomsInputList, priceNormalizeInputList)
        Visualization.build3DRegressionLinearPlusInput(normalizeData, wLast, newPrice, normalizeDataInput)
    else:
        Visualization.build3DRegressionLinear(normalizeData, wLast, newPrice)
    Visualization.build2DInfo(price, newPrice, MSE, lastIteration)
