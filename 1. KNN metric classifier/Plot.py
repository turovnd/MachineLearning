# coding=utf-8
from DatasetProcessing import *

import matplotlib.patches
import matplotlib.pyplot as plt

"""
"""


class Plot(object):
    """Initialization variables"""
    def __init__(self):
        pass

    """Метод визуализации полученных датасетов на графике двумерного простраства (x,y).

    График включает четыре группы точек: две обучающих и две тестирующих выборки.
    Применяется после вызова метода getDataset(t).
    Args:
        trainingDotsWithClass: лист, содержащий датасет обучающих точек в виде ([x,y],classDot).
        testDotsWithClass: лист, содержащий датасет тестирующих точек в виде ([x,y],classDot).

    Returns:
        0: удачное исполнение.
    """
    @staticmethod
    def buildPlotWithAllDots(trainingDotsWithClass, testDotsWithClass):
        trainingDots0 = DatasetProcessing.getDotsByClass(trainingDotsWithClass, 0)
        trainingDots1 = DatasetProcessing.getDotsByClass(trainingDotsWithClass, 1)
        testDots0 = DatasetProcessing.getDotsByClass(testDotsWithClass, 0)
        testDots1 = DatasetProcessing.getDotsByClass(testDotsWithClass, 1)
        colors = ['red', 'blue']
        groupTraining0 = plt.scatter([trainingDots0[i][0] for i in range(len(trainingDots0))],
                                     [trainingDots0[i][1] for i in range(len(trainingDots0))],
                                     color=colors[0])
        groupTraining1 = plt.scatter([trainingDots1[i][0] for i in range(len(trainingDots1))],
                                     [trainingDots1[i][1] for i in range(len(trainingDots1))],
                                     color=colors[1])
        groupTest0 = plt.scatter([testDots0[i][0] for i in range(len(testDots0))],
                                 [testDots0[i][1] for i in range(len(testDots0))],
                                 color=colors[0],
                                 alpha=0.2)
        groupTest1 = plt.scatter([testDots1[i][0] for i in range(len(testDots1))],
                                 [testDots1[i][1] for i in range(len(testDots1))],
                                 color=colors[1],
                                 alpha=0.2)
        # заголовок
        plt.title('Отображение входного датасета')
        # одинаковый размер осей
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        # подпись осей
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        # легенда графика
        plt.legend((groupTraining0, groupTraining1, groupTest0, groupTest1),
                   ('training-0', 'training-1', 'test-0', 'test-1'),
                   loc='upper right')
        plt.show()
        return 0

    """Метод визуализации обучающего датасета и неизвестной точки
     на графике двумерного простраства (x,y) c помощью вычисления центроида.

    Производится формирование датасета точек одного класса в виде ([x,y]).
    Центроид - среднееарифметическое координат (x,y) отдельного класса точек.
    После вычисления центроида происходит вычисление евлидового расстояния.
    Наименьшее расстояние от центроида класса до неизвестной точки означает, что точка
     принадлежит к этому классу.
    Применяется после вызова метода getDataset(t).

    Args:
        trainingDotsWithClass: лист, содержащий датасет обучающих точек в виде ([x,y],classDot).
        testDotsWithClass: лист, содержащий датасет тестирующих точек в виде ([x,y],classDot).
        num: номер тестового значения (частный случай, в общем случае прогоняются все точки
         без графиков).   

    Returns:
        0: удачное исполнение.    
    """
    @staticmethod
    def buildPlotCentroid(trainingDotsWithClass, testDotsWithClass, num):
        colors = ['red', 'blue', 'green']

        if num > len(testDotsWithClass):
            print("error")
            return -1

        trainingDots0 = DatasetProcessing.getDotsByClass(trainingDotsWithClass, 0)
        trainingDots1 = DatasetProcessing.getDotsByClass(trainingDotsWithClass, 1)
        testDot = testDotsWithClass[num]
        groupTraining0 = plt.scatter([trainingDots0[i][0] for i in range(len(trainingDots0))],
                                     [trainingDots0[i][1] for i in range(len(trainingDots0))],
                                     color=colors[0])
        groupTraining1 = plt.scatter([trainingDots1[i][0] for i in range(len(trainingDots1))],
                                     [trainingDots1[i][1] for i in range(len(trainingDots1))],
                                     color=colors[1])
        centroid0 = DatasetProcessing.getCentroid(trainingDots0)
        centroid1 = DatasetProcessing.getCentroid(trainingDots1)
        groupTrainingCenter0 = plt.scatter(centroid0[0], centroid0[1], marker='*', color=colors[0])
        groupTrainingCenter1 = plt.scatter(centroid1[0], centroid1[1], marker='*', color=colors[1])
        unknownDot = plt.scatter(testDot[0][0], testDot[0][1], color=colors[2])
        euclideanDistance0 = DatasetProcessing.computingEuclideanDistance2D(testDot[0], centroid0)
        euclideanDistance1 = DatasetProcessing.computingEuclideanDistance2D(testDot[0], centroid1)

        # тестовые данные
        print('кооридинаты неизвестной точки', testDot[0][0], testDot[0][1])
        print('координаты центра класса 0:', centroid0[0], centroid0[1])
        print('координаты центра класса 1:', centroid1[0], centroid1[1])
        print('из центра 0:', euclideanDistance0)
        print('из центра 1:', euclideanDistance1)
        if euclideanDistance0 < euclideanDistance1:
            print('неизвестная точка принадлежит классу 0')
        else:
            print('неизвестная точка принадлежит классу 1')
        print([testDot[0][0], centroid0[0]], [testDot[0][1], centroid0[1]])

        # построение отрезка, соединяющего центроиды и неизвестную точку
        plt.plot([testDot[0][0], centroid0[0]], [testDot[0][1], centroid0[1]], color=colors[2])
        plt.plot([testDot[0][0], centroid1[0]], [testDot[0][1], centroid1[1]], color=colors[2])
        # заголовок
        plt.title('Отображение обучающей последовательности и неизвестной точки')
        # одинаковый размер осей
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        # подпись осей
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        # легенда графика
        plt.legend((groupTraining0, groupTraining1, unknownDot, groupTrainingCenter0, groupTrainingCenter1),
                   ('training-0', 'training-1', 'unknownDot', 'Center-0', 'Center-1'),
                   loc='upper right')
        plt.show()
        return 0

    """Метод визуализации обучающего датасета и неизвестной точки на графике
     двумерного простраства (x,y) c помощью вычисления наибольшего  количества ближайших соседей.

    Максимальное количество точек одного класса в круге определяет класс тестируемой точки.

    Args:
        trainingDotsWithClass: лист, содержащий датасет обучающих точек в виде ([x,y],classDot).
        testDotsWithClass: лист, содержащий датасет тестирующих точек в виде ([x,y],classDot).
        num: номер тестового значения (частный случай, в общем случае прогоняются все точки
         без графиков).
        k: количество соседей тестируемой точки, для определения её класса;
        metrics: метрика расстояния:
            manhattan - манхэттенское расстояние;
            euclidean - евлидово расстояние.
        coordinateTransformation - пространственное преобразование:
            none - без преобразования;
            elliptic - преобразование эллиптического параболоида;
            hyperbolic - преобразование гиперболического параболоида.
            
    Returns:
        0: удачное исполнение.    
    """
    #TODO ax = Axes3D(fig)
    @staticmethod
    def buildPlotCircle(trainingDotsWithClass, testDotsWithClass, num, k, metrics, coordinateTransformation):
        colors = ['red', 'blue', 'green']

        if num > len(testDotsWithClass):
            print("error")
            return -1

        trainingDots0 = DatasetProcessing.getDotsByClass(trainingDotsWithClass, 0)
        trainingDots1 = DatasetProcessing.getDotsByClass(trainingDotsWithClass, 1)
        groupTraining0 = plt.scatter([trainingDots0[i][0] for i in range(len(trainingDots0))],
                                     [trainingDots0[i][1] for i in range(len(trainingDots0))],
                                     color=colors[0])
        groupTraining1 = plt.scatter([trainingDots1[i][0] for i in range(len(trainingDots1))],
                                     [trainingDots1[i][1] for i in range(len(trainingDots1))],
                                     color=colors[1])
        testDot = testDotsWithClass[num]
        unknownDot = plt.scatter(testDot[0][0], testDot[0][1], color=colors[2])
        testDistance = []

        for i in range(len(trainingDotsWithClass)):
            if coordinateTransformation == "none":
                if metrics == "manhattan":
                    testDistance.append([DatasetProcessing.computingManhattanDistance2D
                                         (testDot[0], trainingDotsWithClass[i][0]), trainingDotsWithClass[i][1]])
                elif metrics == "euclidean":
                    testDistance.append([DatasetProcessing.computingEuclideanDistance2D
                                         (testDot[0], trainingDotsWithClass[i][0]), trainingDotsWithClass[i][1]])
            elif coordinateTransformation == "elliptic" or coordinateTransformation == "hyperbolic":
                if metrics == "manhattan":
                    testDistance.append([DatasetProcessing.computingManhattanDistance3D
                                         (testDot[0], trainingDotsWithClass[i][0]), trainingDotsWithClass[i][1]])
                elif metrics == "euclidean":
                    testDistance.append([DatasetProcessing.computingEuclideanDistance3D
                                         (testDot[0], trainingDotsWithClass[i][0]), trainingDotsWithClass[i][1]])
        # сортировка листа расстояний от меньшего к большему
        n = 1
        while n < len(testDistance):
            for i in range(len(testDistance) - n):
                if testDistance[i][0] > testDistance[i + 1][0]:
                    testDistance[i], testDistance[i + 1] = testDistance[i + 1], testDistance[i]
            n += 1

        class_0 = 0
        class_1 = 0
        for i in range(k - 1):
            if testDistance[i][1] == 1:
                class_1 += 1
            else:
                class_0 += 1

        # # тестовые данные
        # if class_0 > class_1:
        #     print('неизвестная точка принадлежит классу 0')
        # else:
        #     print('неизвестная точка принадлежит классу 1')

        # построение окружности радиуса kой точки с центром неизвестной точки
        circle = matplotlib.patches.Circle(testDot[0], float(testDistance[k - 1][0]), fill=False)
        plt.gca().add_patch(circle)
        # заголовок
        plt.title('k ближайших соседей')
        # одинаковый размер осей
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        # подпись осей
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        # легенда графика
        plt.legend((groupTraining0, groupTraining1, unknownDot),
                   ('training-0', 'training-1', 'unknown dot'),
                   loc='lower left')
        plt.show()
        return 0
