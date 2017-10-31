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
