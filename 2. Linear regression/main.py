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
    kLearningRate = 1.25
    while kLearningRate > 0.01:
        (lastIteration, MSE, wLast, newPrice, weight_hist1, weight_hist2) = \
            GradientDescent.calculateGradientDescent(data, 1, kLearningRate, 3000, 0.0000000001, 1)
        kLearningRate = kLearningRate - 0.01

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
        normalizeDataInput = DatasetProcessing.getCombinedInputData(areaInputList, roomsInputList,
                                                                    priceNormalizeInputList)
        Visualization.build3DRegressionLinearPlusInput(normalizeData, wLast, newPrice, normalizeDataInput)
    else:
        Visualization.build3DRegressionLinear(normalizeData, wLast, newPrice)
    Visualization.build2DInfo(price, newPrice, MSE, lastIteration)
