from DatasetProcessing import *
# import matplotlib.patches
import matplotlib.pyplot as plt
import pylab
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
    def buildStartDatasetPlot(data):
        x, y, z = DatasetProcessing.getSeparetedData(data)
        fig = plt.figure()
        axes = Axes3D(fig)
        axes.scatter(x, y, z, color="#00CC00")
        # заголовок
        plt.title("Отображение входного датасета")
        # подпись осей
        axes.set_xlabel('Area')
        axes.set_ylabel('Rooms')
        axes.set_zlabel('Price')
        plt.show()
        return 0

    @staticmethod
    def buildNewDatasetPlot(w, data, newZ, w0, w1, MSE, newPrice):
        x, y, z = DatasetProcessing.getSeparetedData(data)
        fig = plt.figure()
        axes = Axes3D(fig)
        axes.scatter(x, y, z, color="#00CC00")
        axes.scatter(x, y, newZ, color="#FF0000")
        # axes.scatter(x, y, w0, color="#FFCC00")
        # axes.plot_wireframe(x, newZ, newPrice, color="#FFCC00")
        # заголовок
        plt.title("Отображение обновленного? датасета")
        # подпись осей
        axes.set_xlabel('Area')
        axes.set_ylabel('Rooms')
        axes.set_zlabel('Price')
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
        plt.title("Изменение среднеквадратичного отклонения с каждой итерацией")
        plt.xlabel("$iteration$")
        plt.ylabel("$J(w) - MSE$")
        plt.grid(True)
        plt.subplot(212)
        colors = ["#00B945", "#FF2C00"]
        linewidths = [4, 4]
        labels = ["predicted price", "given price"]
        plt.plot([newPrice[0][i] for i in range(len(newPrice[0]))], color=colors[0],
                 linewidth=linewidths[0], label=labels[0])
        plt.plot([price[i] for i in range(len(price))], color=colors[1],
                 linewidth=linewidths[1], label=labels[1])
        # заголовок
        plt.title("Визуализация данных и вычисленных цен")
        # подпись осей
        plt.xlabel("$iteration$")
        plt.ylabel("$price$")
        # легенда графика
        plt.legend(loc=1, fontsize="small")
        plt.grid(True)
        plt.show()
        return 0
