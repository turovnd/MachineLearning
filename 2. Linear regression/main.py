from DatasetProcessing import *
from Visualization import *
from Evolution import *

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

    # gradient
    table = []
    print(tabulate(table,
                   headers=["breakCriterion", "lastIteration",
                            "stepsNumber", "kLearningRate",
                            "current alphaLearningRate",
                            "epsilonLimitation", "errorAvg(MSE)",
                            "MSE[i]", "MSE[i-1]", "MSE[0]", "abs(MSE[i] - MSE[i-1])",
                            "weight_NP[1]", "weight_NP[2]", "abs(weight_NP[1] - weight_NP[2]"],
                   tablefmt='orgtbl'))
    kLearningRate = 1.61
    # while kLearningRate > 1.00:
    (lastIteration, MSEGradient, wLast, newPriceGradient, weight_hist1, weight_hist2) = \
            GradientDescent.calculateGradientDescent(data, 0, kLearningRate, 50000, 0.0000000001, 1)
        # kLearningRate = kLearningRate - 0.1

    # gradient input
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

    # evolution
    numberOfIterations = 60000
    MSEEvolutionTop0, MSEEvolutionTop1, MSEEvolutionTop2, MSE_hist0, MSE_hist1, MSE_hist2, MSE_hist3, MSE_hist4, \
        MSE_hist5, MSE_hist6, wTopEvo = Evolution.startEvolution(numberOfIterations, 1, 100)

    # newPriceEvo
    x0 = np.ones(len(area))
    XTranspose_NP = np.vstack((x0, area, rooms))  # двумерный массив [,.....,],[,.....,]
    X_NP = np.transpose(XTranspose_NP)
    newPriceEvo = X_NP.dot(wTopEvo)
    newPriceEvo = newPriceEvo.tolist()

    # visualization
    Visualization.build3DStartDataset(data)
    Visualization.build3DCostFunction(weight_hist1, weight_hist2, MSEGradient, lastIteration)
    if (len(areaInputList) != 0):
        priceNormalizeInputList = GradientDescent.calculateInputPrice(areaInputList, roomsInputList, wLast)
        normalizeDataInput = DatasetProcessing.getCombinedInputData(areaInputList, roomsInputList,
                                                                    priceNormalizeInputList)
        Visualization.build3DRegressionLinearPlusInput(normalizeData, wLast, newPriceGradient, normalizeDataInput)
    else:
        Visualization.build3DRegressionLinear(normalizeData, wLast, newPriceGradient)
    Visualization.build3DRegressionLinear(normalizeData, wTopEvo, newPriceEvo)
    Visualization.build2DIndividualMSEEvolution(MSE_hist0, MSE_hist1, MSE_hist2, MSE_hist3,
                                                MSE_hist4, MSE_hist5, MSE_hist6, numberOfIterations)

    Visualization.build2DInfo(price, newPriceGradient, newPriceEvo, MSEGradient, MSEEvolutionTop0, MSEEvolutionTop1,
                              MSEEvolutionTop2, lastIteration)
    Visualization.build3DRegressionLinearGradientVsEvolution(normalizeData, wLast, newPriceGradient,
                                                             wTopEvo, newPriceEvo)