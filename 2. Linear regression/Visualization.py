from DatasetProcessing import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
"""
"""


class Visualization(object):
    """Initialization variables"""
    def __init__(self):
        pass

    """Метод визуализации полученного датасета на графике трехмерного простраства (x,y,z)=(area, rooms, price).
        
    Args: 
        data: лист, содержащий входной датасет в виде (area,rooms,price).

    Returns:
        0: удачное исполнение.
    """
    @staticmethod
    def build3DStartDataset(data):
        x, y, z = DatasetProcessing.getSeparetedData(data)
        fig = plt.figure()
        axes = Axes3D(fig)
        axes.scatter(x, y, z, color="#00CC00")
        # title
        plt.title("Visualization started dataset")
        # label
        axes.set_xlabel('Area')
        axes.set_ylabel('Rooms')
        axes.set_zlabel('Price')
        plt.show()
        return 0

    """Метод отображения регрессионной плоскости относительно нормализованного датасета.

    Args: 
        normalizeData: лист, содержащий нормализованный датасет в виде (area,rooms,price).
        w, лист, содержащий веса w0 для x0, w1 для x1, w2 для x2.
        
    Returns:
        0: удачное исполнение.
    """
    @staticmethod
    def build3DRegressionLinear(normalizeData, w, zNew):
        # for scatter
        xNormalizeData, yNormalizeData, zNormalizeData = DatasetProcessing.getSeparetedData(normalizeData)
        updateX = []
        updateY = []
        for i in range(len(xNormalizeData)):
            updateX.append(xNormalizeData[i] * w[1][0])
            updateY.append(yNormalizeData[i] * w[2][0])

        # for plot_wireframe
        n = 30  # плотность сетки
        # x -2 4 # y -3 2 # z -4 6
        X = np.linspace(-2, 4, n)
        Y = np.linspace(-3, 2, n)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Z[i, j] = w[0][0] + X[i, j] * w[1][0] + Y[i, j] * w[2][0]

        # parameters for plots
        colors = ["#00CC00", "#FF0000", "#560EAD"]
        linewidths = [4, 6]
        labels = ["started normalize dataset", "calculated normalize dataset", "calculated regression plane"]

        fig = plt.figure()
        ax1 = Axes3D(fig)
        ax1.scatter(xNormalizeData, yNormalizeData, zNormalizeData, color=colors[0], linewidth=linewidths[0],
                    label=labels[0])
        ax1.scatter(xNormalizeData, yNormalizeData, zNew, color=colors[1], linewidth=linewidths[1],
                    label=labels[1])
        ax1.plot_wireframe(X, Y, Z, color=colors[2], label=labels[2])
        # title
        plt.title("Regression plane relatively datasets")
        # label
        ax1.set_xlabel("Area normalize")
        ax1.set_ylabel("Rooms normalize")
        ax1.set_zlabel("Price normalize")
        # legend
        plt.legend(loc=3, fontsize="small")
        plt.show()
        return 0

    """Метод отображения нескольких 2D графиков за один вызов.
        1) Изменение среднеквадратичного отклонения с каждой итерацией.
        2) Визуализация данных и вычисленных цен.
        
    Args: 
        price: лист, содержащий цену из датасета.
        newPrice: лист, содержащий пересчитанную цену.
        MSE: лист, содержащий среднеквадртичные отклоения.
        lastIteration: последняя иттерация вычислений.
        
    Returns:
        0: удачное исполнение.
    """
    @staticmethod
    def build2DInfo(price, newPrice, MSE, lastIteration):
        plt.subplot(211)
        plt.plot([MSE[i] for i in range(lastIteration)], color="#560EAD", linewidth=4)
        # title
        plt.title("Measurement of MSE with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$J(w) - MSE$")
        plt.grid(True)

        plt.subplot(212)
        colors = ["#00B945", "#FF2C00"]
        linewidths = [4, 4]
        labels = ["calculated price", "started price"]
        plt.plot([newPrice[0][i] for i in range(len(newPrice[0]))], color=colors[0], linewidth=linewidths[0],
                 label=labels[0])
        plt.plot([price[i] for i in range(len(price))], color=colors[1], linewidth=linewidths[1],
                 label=labels[1])
        # title
        plt.title("Visualization started and calculated prices")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$price$")
        # legend
        plt.legend(loc=1, fontsize="small")
        plt.grid(True)
        plt.show()
        return 0
