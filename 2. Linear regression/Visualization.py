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
    def build3DRegressionLinearGradientVsEvolution(normalizeData, wGradient, priceGradient, wEvolution, priceEvolution):
        """Метод отображения регрессионной плоскости относительно нормализованного датасета.

        Args:
            normalizeData: лист, содержащий нормализованный датасет в виде (area,rooms,price).
            wGradient, лист, содержащий последние веса w0 для x0, w1 для x1, w2 для x2 для градиентного спуска.
            priceGradient: лист, содержащий рассчитанные цены для градиентного спуска.
            wEvolution, лист, содержащий последние веса w0 для x0, w1 для x1, w2 для x2 для эволюции.
            priceEvolution: лист, содержащий рассчитанные цены для эволюции.
        Returns:
            0: удачное исполнение.
        """
        # for scatter
        xNormalizeData, yNormalizeData, zNormalizeData = DatasetProcessing.getSeparetedData(normalizeData)

        # for plot_wireframe
        n = 30  # плотность сетки
        # x -2 4 # y -3 2 # z -4 6
        X = np.linspace(-4, 6, n)
        Y = np.linspace(-5, 4, n)
        X, Y = np.meshgrid(X, Y)
        ZGradient = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ZGradient[i, j] = wGradient[0][0] + X[i, j] * wGradient[1][0] + Y[i, j] * wGradient[2][0]
        ZEvolution = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ZEvolution[i, j] = wEvolution[0][0] + X[i, j] * wEvolution[1][0] + Y[i, j] * wEvolution[2][0]
        # parameters for plots
        colors = ["#00CC00", "#AD560E", "#AD780E", "#153174", "#086868"]
        linewidths = [4, 6]
        labels = ["started normalize dataset", "gradient calculated normalize dataset",
                  "gradient calculated regression plane", "evolution calculated normalize dataset",
                  "evolution calculated regression plane"]

        fig = plt.figure()
        ax1 = Axes3D(fig)
        ax1.scatter(xNormalizeData, yNormalizeData, zNormalizeData, color=colors[0], linewidth=linewidths[0],
                    label=labels[0])
        # gradient
        ax1.scatter(xNormalizeData, yNormalizeData, priceGradient, color=colors[1], linewidth=linewidths[0],
                    label=labels[1])
        ax1.plot_wireframe(X, Y, ZGradient, color=colors[2], label=labels[2])
        # evolution
        ax1.scatter(xNormalizeData, yNormalizeData, priceEvolution, color=colors[3], linewidth=linewidths[0],
                    label=labels[3])
        ax1.plot_wireframe(X, Y, ZEvolution, color=colors[4], label=labels[4])
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
    def build2DInfo(price, priceGradient, priceEvolution, MSEGradient, MSEEvolutionTop0, MSEEvolutionTop1,
                    MSEEvolutionTop2, lastIteration):
        """Метод отображения нескольких 2D графиков за один вызов.
                1) Изменение среднеквадратичного отклонения с каждой итерацией.
                2) Визуализация данных и вычисленных цен.

        Args:
            price: лист, содержащий цену из датасета.
            priceGradient: лист, содержащий рассчитанные цены для градиентного спуска.
            priceEvolution: лист, содержащий рассчитанные цены для эволюции.
            MSEGradient: лист, содержащий среднеквадртичные отклонения градентного спуска.
            MSEEvolutionTop0: лист, первая лучшие среднеквадртичные отклонения эволюции.
            MSEEvolutionTop1: лист, вторые лучшие среднеквадртичные отклонения эволюции.
            MSEEvolutionTop2: лист, третьи лучшие среднеквадртичные отклонения эволюции.
            lastIteration: последняя иттерация вычислений.


        Returns:
            0: удачное исполнение.
        """
        plt.subplot(211)
        colors = ["#0A7748", "#104670", "#AE6A0F", "#9B0D38"]
        linewidths = [3, 3, 3, 3]
        labels = ["MSEGradient", "MSEEvolutionTop0", "MSEEvolutionTop1", "MSEEvolutionTop2"]
        plt.plot([MSEGradient[i] for i in range(lastIteration)], color=colors[0], linewidth=linewidths[0],
                 label=labels[0])
        plt.plot([MSEEvolutionTop0[i] for i in range(len(MSEEvolutionTop0))], color=colors[1], linewidth=linewidths[1],
                 label=labels[1])
        plt.plot([MSEEvolutionTop1[i] for i in range(len(MSEEvolutionTop0))], color=colors[2], linewidth=linewidths[2],
                 label=labels[2])
        plt.plot([MSEEvolutionTop2[i] for i in range(len(MSEEvolutionTop0))], color=colors[3], linewidth=linewidths[3],
                 label=labels[3])
        # title
        plt.title("Measurement of MSE with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
        plt.legend(loc=1, fontsize="small")
        plt.grid(True)

        plt.subplot(212)
        colors = ["#00B945", "#FF2C00", "#AA2C00"]
        linewidths = [2, 2, 2]
        labels = ["gradient calculated price", "evolution calculated price", "started price"]
        plt.plot([priceGradient[0][i] for i in range(len(priceGradient[0]))], color=colors[0], linewidth=linewidths[0],
                 label=labels[0])
        plt.plot([priceEvolution[i][0] for i in range(len(priceEvolution))], color=colors[1],
                 linewidth=linewidths[1], label=labels[1])
        plt.plot([price[i] for i in range(len(price))], color=colors[2], linewidth=linewidths[2],
                 label=labels[2])
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
    def build2DIndividualMSEEvolution(MSE0, MSE1, MSE2, MSE3, MSE4, MSE5, MSE6, lastIteration):
        """Метод отображения изменения MSE особей в ходе эволюции.

        Args:
            MSE0: лист, изменение ошибки особи 0.
            MSE1: лист, изменение ошибки особи 1.
            MSE2: лист, изменение ошибки особи 2.
            MSE3: лист, изменение ошибки особи 3.
            MSE4: лист, изменение ошибки особи 4.
            MSE5: лист, изменение ошибки особи 5.
            MSE6: лист, изменение ошибки особи 6.
            lastIteration: последняя иттерация вычислений.

        Returns:
            0: удачное исполнение.
        """
        plt.subplot(241)
        plt.plot([MSE0[i] for i in range(lastIteration)], color="#AA0000", linewidth=1)
        # title
        plt.title("Measurement of MSE0 with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
        plt.grid(True)

        plt.subplot(242)
        plt.plot([MSE1[i] for i in range(lastIteration)], color="#AA0000", linewidth=1)
        # title
        plt.title("Measurement of MSE1 with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
        plt.grid(True)

        plt.subplot(243)
        plt.plot([MSE2[i] for i in range(lastIteration)], color="#AA0000", linewidth=1)
        # title
        plt.title("Measurement of MSE2 with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
        plt.grid(True)

        plt.subplot(244)
        plt.plot([MSE3[i] for i in range(lastIteration)], color="#AA0000", linewidth=1)
        # title
        plt.title("Measurement of MSE3 with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
        plt.grid(True)

        plt.subplot(245)
        plt.plot([MSE4[i] for i in range(lastIteration)], color="#AA0000", linewidth=1)
        # title
        plt.title("Measurement of MSE4 with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
        plt.grid(True)

        plt.subplot(246)
        plt.plot([MSE5[i] for i in range(lastIteration)], color="#AA0000", linewidth=1)
        # title
        plt.title("Measurement of MSE5 with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
        plt.grid(True)

        plt.subplot(247)
        plt.plot([MSE6[i] for i in range(lastIteration)], color="#AA0000", linewidth=1)
        # title
        plt.title("Measurement of MSE6 with each iteration")
        # label
        plt.xlabel("$iteration$")
        plt.ylabel("$MSE$")
        plt.grid(True)
        plt.show()
        return 0
