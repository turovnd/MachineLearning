from DatasetProcessing import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
"""
"""


class Visualization(object):
    """Initialization variables"""
    def __init__(self):
        pass

    @staticmethod
    def build3DStartDataset(data):
        """Метод отображения полученного датасета на графике трехмерного простраства (x,y,z)=(area, rooms, price).

        Args:
            data: лист, содержащий входной датасет в виде (area,rooms,price).

        Returns:
            0: удачное исполнение.
        """
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

    @staticmethod
    def build3DRegressionLinearPlusInput(normalizeData, wLast, zNew, normalizeDataInput):
        """Метод отображения регрессионной плоскости относительно нормализованного датасета.

        Args:
            normalizeData: лист, содержащий нормализованный датасет в виде (area,rooms,price).
            wLast, лист, содержащий последние веса w0 для x0, w1 для x1, w2 для x2.
            zNew: лист, содержащий рассчитанные цены.
            normalizeDataInput: лист, содержащий введенные объединненные нормализованне датасеты
             в виде (areaNormalizeInputList,roomsNormalizeInputList,priceNormalizeInputList).

        Returns:
            0: удачное исполнение.
        """
        # for scatter
        xNormalizeData, yNormalizeData, zNormalizeData = DatasetProcessing.getSeparetedData(normalizeData)
        xInputNormalizeData, yInputNormalizeData, zInputNormalizeData = \
            DatasetProcessing.getSeparetedData(normalizeDataInput)

        # for plot_wireframe
        n = 30  # плотность сетки
        # x -2 4 # y -3 2 # z -4 6
        X = np.linspace(-2, 4, n)
        Y = np.linspace(-3, 2, n)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Z[i, j] = wLast[0][0] + X[i, j] * wLast[1][0] + Y[i, j] * wLast[2][0]

        # parameters for plots
        colors = ["#00CC00", "#FF0000", "#560EAD", "#FF9500"]
        linewidths = [4, 6]
        labels = ["started normalize dataset", "calculated normalize dataset", "calculated regression plane",
                  "input calculated normalize dataset"]

        fig = plt.figure()
        ax1 = Axes3D(fig)
        ax1.scatter(xNormalizeData, yNormalizeData, zNormalizeData, color=colors[0], linewidth=linewidths[0],
                    label=labels[0])
        ax1.scatter(xNormalizeData, yNormalizeData, zNew, color=colors[1], linewidth=linewidths[1],
                    label=labels[1])
        ax1.scatter(xInputNormalizeData, yInputNormalizeData, zInputNormalizeData, color=colors[3], linewidth=linewidths[1],
                    label=labels[3])
        ax1.plot_wireframe(X, Y, Z, color=colors[2], label=labels[2], alpha=.5)
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

    @staticmethod
    def build3DRegressionLinear(normalizeData, wLast, zNew):
        """Метод отображения регрессионной плоскости относительно нормализованного датасета.

        Args:
            normalizeData: лист, содержащий нормализованный датасет в виде (area,rooms,price).
            wLast, лист, содержащий последние веса w0 для x0, w1 для x1, w2 для x2.
            zNew: лист, содержащий рассчитанные цены.

        Returns:
            0: удачное исполнение.
        """
        # for scatter
        xNormalizeData, yNormalizeData, zNormalizeData = DatasetProcessing.getSeparetedData(normalizeData)

        # for plot_wireframe
        n = 30  # плотность сетки
        # x -2 4 # y -3 2 # z -4 6
        X = np.linspace(-2, 4, n)
        Y = np.linspace(-3, 2, n)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Z[i, j] = wLast[0][0] + X[i, j] * wLast[1][0] + Y[i, j] * wLast[2][0]

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

    @staticmethod
    def build3DCostFunction(weight_hist1, weight_hist2, MSE, lastIteration):
        """Метод отображения графика функции стоимости и точек MSE первого вычисления и последнего.

        Args:
            weight_hist1(): массив numpy, содержащий веса всех итераций для w1.
            weight_hist2(): массив numpy, содержащий веса всех итераций для w2.
            MSE: лист, содержащий среднеквадртичные отклоения.
            lastIteration: число, последняя иттерация вычислений.

        Returns:
            0: удачное исполнение.
        """
        # for plot_wireframe
        n = 20  # плотность сетки
        # x -2 4 # y -3 2 # z -4 6
        X = np.linspace(-max(MSE), max(MSE), n)
        Y = np.linspace(-max(MSE), max(MSE), n)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Z[i, j] = (X[i, j] * X[i, j] * weight_hist1[i] + Y[i, j] * Y[i, j] *
                           weight_hist2[i])

        # parameters for plots
        colors = ["#00CC00", "#FFE400", "#560EAD"]
        linewidths = [10, 10]
        labels = ["started calculate MSE", "ended calculate MSE"]
        fig = plt.figure()
        ax1 = Axes3D(fig)
        ax1.plot_surface(X, Y, Z, cmap=cm.rainbow, alpha=.3)
        ax1.scatter(weight_hist1[0], weight_hist2[0], MSE[0], color=colors[0], linewidth=linewidths[1],
                    label=labels[0])
        ax1.scatter(weight_hist1[lastIteration], weight_hist2[lastIteration], MSE[lastIteration],
                    color=colors[1], linewidth=linewidths[1], label=labels[1])
        # title
        plt.title("Cost function for all values weights x1, x2 parameters")
        # label
        ax1.set_xlabel("weight_hist1")
        ax1.set_ylabel("weight_hist2")
        ax1.set_zlabel("MSE")
        # legend
        plt.legend(loc=3, fontsize="small")
        plt.show()
        return 0

    @staticmethod
    def build2DInfo(price, newPrice, MSE, lastIteration):
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
        plt.subplot(211)
        plt.plot([MSE[i] for i in range(lastIteration)], color="#560EAD", linewidth=4)
        # title
        plt.title("Measurement of MSE with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
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
        plt.xlabel("number")
        plt.ylabel("$price$")
        # legend
        plt.legend(loc=1, fontsize="small")
        plt.grid(True)
        plt.show()
        return 0

    @staticmethod
    def build2DMSEEvolution(MSE0, MSE1, MSE2, lastIteration):
        """Метод отображения изменения MSE в ходе эволюции.

        Args:
            MSE0: число, лучшая ошибка 0 особи.
            MSE1: число, лучшая ошибка 1 особи.
            MSE2: число, лучшая ошибка 2 особи.
            # MSE1: число, ошибка особи 1.
            # MSE2: число, ошибка особи 2.
            # MSE3: число, ошибка особи 3.
            # MSE4: число, ошибка особи 4.
            # MSE5: число, ошибка особи 5.
            # MSE6: число, ошибка особи 6.
            lastIteration: последняя иттерация вычислений.

        Returns:
            0: удачное исполнение.
        """
        plt.subplot(221)
        plt.plot([MSE0[i] for i in range(lastIteration)], color="#FF0000", linewidth=1)
        # title
        plt.title("Measurement of MSE top0 with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
        plt.grid(True)

        plt.subplot(222)
        plt.plot([MSE1[i] for i in range(lastIteration)], color="#FF00FF", linewidth=1)
        # title
        plt.title("Measurement of MSE top1 with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
        plt.grid(True)

        plt.subplot(223)
        plt.plot([MSE2[i] for i in range(lastIteration)], color="#00FF00", linewidth=1)
        # title
        plt.title("Measurement of MSE top2 with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
        plt.grid(True)

        #
        # plt.subplot(212)
        # colors = ["#00B945", "#FF2C00"]
        # linewidths = [4, 4]
        # labels = ["calculated price", "started price"]
        # plt.plot([newPrice[0][i] for i in range(len(newPrice[0]))], color=colors[0], linewidth=linewidths[0],
        #          label=labels[0])
        # plt.plot([price[i] for i in range(len(price))], color=colors[1], linewidth=linewidths[1],
        #          label=labels[1])
        # # title
        # plt.title("Visualization started and calculated prices")
        # # label
        # plt.xlabel("number")
        # plt.ylabel("$price$")
        # # legend
        # plt.legend(loc=1, fontsize="small")
        # plt.grid(True)
        plt.show()
        return 0
