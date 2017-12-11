# coding=utf-8
import math
import random

"""
"""


class DatasetProcessing(object):
    """Initialization variables"""
    def __init__(self):
        pass

    """Метод обработки входного датасета.

    file 'dataset.txt': входной датасет, содержащий информацию о точках в виде (x,y,classDot).
    z = (x^2/a^2)+(y^2/b^2); a=1; b=1 (случае пространственного преобразования).
    
    Args:
        filename: имя входного датасета.
        coordinateTransformation - пространственное преобразование:
            none - без преобразования;
            elliptic - преобразование эллиптического параболоида;
            hyperbolic - преобразование гиперболического параболоида.
            
    Returns:
        data: лист, содержащий входной датасет в виде ([x,y],classDot) или ([x,y,z],classDot) в зависимости от
         coordinateTransformation.
    """
    @staticmethod
    def getDataset(filename, coordinateTransformation):
        data = []
        f = open(filename)
        for line in f:
            dot_x, dot_y, dot_class = line.split(',')
            if coordinateTransformation == "none":
                data.append([[float(dot_x), float(dot_y)], int(dot_class)])
            elif coordinateTransformation == "elliptic":
                dot_z = math.pow(float(dot_x), 2) + math.pow(float(dot_y), 2)
                data.append([[float(dot_x), float(dot_y), float(dot_z)], int(dot_class)])
            elif coordinateTransformation == "hyperbolic":
                dot_z = math.pow(float(dot_x), 2) - math.pow(float(dot_y), 2)
                data.append([[float(dot_x), float(dot_y), float(dot_z)], int(dot_class)])
        f.close()
        random.shuffle(data)
        return data

    """Метод перемещивания входного массива.
    
    Args:
        number_trainDots: количество тренировочных точек.
        data: лист, содержащий входной датасет в виде ([x,y],classDot) или ([x,y,z],classDot) в зависимости от
         coordinateTransformation.
            
    Returns:
        trainDots: лист, содержащий тренировочные точки в виде ([x,y],classDot) или ([x,y,z],classDot) в зависимости от
         coordinateTransformation.
        testDots: лист, содержащий тестирующие точки в виде ([x,y],classDot) или ([x,y,z],classDot) в зависимости от
         coordinateTransformation.
    """
    @staticmethod
    def getTrainTestDots(number_trainDots, data):
        random.shuffle(data)
        return data[0:number_trainDots], data[number_trainDots + 1:len(data)]

    """Метод формирования датасета точек одного класса в виде ([x,y]).
    
    Args:
        DotsWithClass: лист, содержащий датасет точек в виде ([x,y],classDot).
        classDot: номер класса датасета точек.
    
    Returns:
        dots: лист, содержащий датасет точек в виде ([x,y]).
    """
    @staticmethod
    def getDotsByClass(DotsWithClass, classDot):
        dots = []
        for i in range(len(DotsWithClass)):
            if DotsWithClass[i][1] == classDot:
                dots.append(DotsWithClass[i][0])
        return dots

    """Метод расчета манхэттенского расстояния для двух точек двумерного пространства (x,y).
        Формула: p(x,y) = abs(x0[i]-x1[i])+abs(y0[i]-y1[i]).

        Args:
            trainingDot: совокупность координат (x,y) обучающей точки (из обучающей выборки).
            unknownDot:  совокупность координат (x,y) неизвестной точки (из тестирующей выборки).

        Returns:
            Действительное число, манхэттенское расстояние между двумя точками.
        """
    @staticmethod
    def computingManhattanDistance2D(trainingDot, unknownDot):
        return abs(trainingDot[0] - unknownDot[0]) + abs(trainingDot[1] - unknownDot[1])

    """Метод расчета манхэттенского расстояния для двух точек трехмерного пространства (x,y,z).
    Формула: p(x,y) = abs(x0[i]-x1[i])+abs(y0[i]-y1[i])+abs(z0[i]-z1[i]).

    Args:
        trainingDot: совокупность координат (x,y,z) обучающей точки (из обучающей выборки).
        unknownDot:  совокупность координат (x,y,z) неизвестной точки (из тестирующей выборки).

    Returns:
        Действительное число, манхэттенское расстояние между двумя точками.
    """
    @staticmethod
    def computingManhattanDistance3D(trainingDot, unknownDot):
        return (abs(trainingDot[0] - unknownDot[0]) + abs(trainingDot[1] - unknownDot[1])
                + abs(trainingDot[2] - unknownDot[2]))

    """Метод расчета евлидового расстояния для двух точек двумерного пространства (x,y).
    Формула: p(x,y) = sqrt((x0[i]-x1[i])^2)+(y0[i]-y1[i])^2).
    
    Args:
        trainingDot: совокупность координат (x,y) обучающей точки (из обучающей выборки).
        unknownDot:  совокупность координат (x,y) неизвестной точки (из тестирующей выборки).
    
    Returns:
        Действительное число, евклидово расстояние между двумя точками.
    """
    @staticmethod
    def computingEuclideanDistance2D(trainingDot, unknownDot):
        return math.sqrt(math.pow(trainingDot[0] - unknownDot[0], 2)
                         + math.pow(trainingDot[1] - unknownDot[1], 2))

    """Метод расчета евлидового расстояния для двух точек трехмерного пространства (x,y,z).
        Формула: p(x,y) = sqrt((x0[i]-x1[i])^2)+(y0[i]-y1[i])^2+(z0[i]-z1[i])^2).

        Args:
            trainingDot: совокупность координат (x,y,z) обучающей точки (из обучающей выборки).
            unknownDot:  совокупность координат (x,y,z) неизвестной точки (из тестирующей выборки).

        Returns:
            Действительное число, евклидово расстояние между тремя точками.
        """
    @staticmethod
    def computingEuclideanDistance3D(trainingDot, unknownDot):
        return math.sqrt(math.pow(trainingDot[0] - unknownDot[0], 2) + math.pow(trainingDot[1] - unknownDot[1], 2)
                         + math.pow(trainingDot[2] - unknownDot[2], 2))

    """Метод расчета центрода класса точек двумерного пространства (x,y).
    Формула: sum(x[i]/len(x)).
    
    Args:
        trainingDots: лист, содержащий датасет обучающих точек в виде ([x,y]).
    
    Returns: 
        Лист, содержащий координату центра класса в виде (x,y).
    """
    @staticmethod
    def getCentroid(trainingDots):
        x = sum([trainingDots[i][0] for i in range(len(trainingDots))])
        y = sum([trainingDots[i][1] for i in range(len(trainingDots))])
        return [x / len(trainingDots), y / len(trainingDots)]

    """Метод определения класса тестируемой точки с помощью центроидов.
    
    Args:
        trainingDot0: лист, содержащий датасет обучающих точек класса 0 в виде ([x,y]).
        trainingDot1: лист, содержащий датасет обучающих точек класса 1 в виде ([x,y]).
        unknownDot:  совокупность координат (x,y) неизвестной точки (из тестирующей выборки).
    
    Returns:
        0: неизвестная точка принадлежит классу 0.    
        1: неизвестная точка принадлежит классу 1.    
    """
    @staticmethod
    def classifyDotCentroid(trainingDot0, trainingDot1, unknownDot):
        centerDots0 = DatasetProcessing.getCentroid(trainingDot0)
        centerDots1 = DatasetProcessing.getCentroid(trainingDot1)
        euclideanDistance0 = DatasetProcessing.computingEuclideanDistance2D(unknownDot, centerDots0)
        euclideanDistance1 = DatasetProcessing.computingEuclideanDistance2D(unknownDot, centerDots1)
        if euclideanDistance0 < euclideanDistance1:
            return 0
        else:
            return 1

    """Метод определения класса тестируемой точки с помощью вычисления наибольшего 
     количества ближайших соседей.
    
    Args:
        trainingDotsWithClass: лист, содержащий датасет обучающих точек в виде ([x,y],classDot).
        testDotsWithClass: лист, содержащий датасет тестирующих точек в виде ([x,y],classDot).
        k: количество соседей тестируемой точки, для определения её класса.
        kernelFunction: ядро используемой функции:
            none - без ядра;
            gaussian - нормальное распределение (gaussian);
            logistic - логистическое распределение (logistic).
        metrics: метрика расстояния:
            manhattan - манхэттенское расстояние;
            euclidean - евлидово расстояние.
        coordinateTransformation - пространственное преобразование:
            none - без преобразования;
            elliptic - преобразование эллиптического параболоида;
            hyperbolic - преобразование гиперболического параболоида.
            
    Returns:
        0: неизвестная точка принадлежит классу 0.    
        1: неизвестная точка принадлежит классу 1.
    """
    @staticmethod
    def classifyDotCircle(trainingDotsWithClass, testDotsWithClass, k, kernelFunction, metrics,
                          coordinateTransformation):
        testDist = []
        for i in range(len(trainingDotsWithClass)):
            if coordinateTransformation == "none":
                if metrics == "manhattan":
                    testDist.append(
                        [DatasetProcessing.computingManhattanDistance2D(trainingDotsWithClass[i][0], testDotsWithClass),
                         trainingDotsWithClass[i][1]])
                elif metrics == "euclidean":
                    testDist.append(
                        [DatasetProcessing.computingEuclideanDistance2D(trainingDotsWithClass[i][0], testDotsWithClass),
                         trainingDotsWithClass[i][1]])
            elif coordinateTransformation == "elliptic" or coordinateTransformation == "hyperbolic":
                if metrics == "manhattan":
                    testDist.append(
                        [DatasetProcessing.computingManhattanDistance3D(trainingDotsWithClass[i][0], testDotsWithClass),
                         trainingDotsWithClass[i][1]])
                elif metrics == "euclidean":
                    testDist.append(
                        [DatasetProcessing.computingEuclideanDistance3D(trainingDotsWithClass[i][0], testDotsWithClass),
                         trainingDotsWithClass[i][1]])

        n = 1
        while n < len(testDist):
            for i in range(len(testDist) - n):
                if testDist[i][0] > testDist[i + 1][0]:
                    testDist[i], testDist[i + 1] = testDist[i + 1], testDist[i]
            n += 1

        if kernelFunction == "none":
            class_0 = 0
            class_1 = 0
            for i in range(k):
                if testDist[i][1] == 1:
                    class_1 += 1
                else:
                    class_0 += 1
        elif kernelFunction == "gaussian":
            class_0 = 0
            class_1 = 0
            for i in range(k):
                if testDist[i][1] == 1:
                    class_1 = 1 / (math.sqrt(2 * math.pi) * math.exp(-1 / 2 * math.pow(class_1, 2)))
                else:
                    class_0 = 1 / (math.sqrt(2 * math.pi) * math.exp(-1 / 2 * math.pow(class_0, 2)))
        elif kernelFunction == "logistic":
            class_0 = 0
            class_1 = 0
            for i in range(k):
                if testDist[i][1] == 1:
                    class_1 = 1 / (math.exp(class_1) + 2 + math.exp(-class_1))
                else:
                    class_0 = 1 / (math.exp(class_0) + 2 + math.exp(-class_0))

        if class_0 > class_1:
            return 0
        else:
            return 1

    """#TODO в отдельный файл (кусок не обязателен).Метод формирования датасетов классов
     неизвестных точек с помощью центроидов.
    
    Args:
        trainingDots: совокупность координат (x,y) обучающей точки (из обучающей выборки).
        unknownDots:  совокупность координат (x,y) неизвестной точки (из тестирующей выборки).
    
    Returns:
        testClasses: лист, содержащий датасет с классами неизвестных точек, определенных в алгоритме.
    """
    @staticmethod
    def classifyKNNCentroid(trainingDots, unknownDots):
        trainingDots0 = DatasetProcessing.getDotsByClass(trainingDots, 0)
        trainingDots1 = DatasetProcessing.getDotsByClass(trainingDots, 1)
        testClasses = []
        for i in range(len(unknownDots)):
            dot_class = DatasetProcessing.classifyDotCentroid(trainingDots0, trainingDots1, unknownDots[i][0])
            if dot_class == 0:
                trainingDots0.append(unknownDots[i][0])
            else:
                trainingDots1.append(unknownDots[i][0])
            testClasses.append(dot_class)
        return testClasses

    """Метод формирования датасетов классов неизвестных точек с помощью вычисления наибольшего 
     количества ближайших соседей.
    
    Args:
        trainingDots: совокупность координат (x,y) обучающей точки (из обучающей выборки).
        unknownDots:  совокупность координат (x,y) неизвестной точки (из тестирующей выборки).
        k: количество соседей тестируемой точки, для определения её класса.
        kernelFunction: ядро используемой функции.
        metrics: метрика расстояния:
                manhattan - манхэттенское расстояние;
                euclidean - евлидово расстояние.
        coordinateTransformation - пространственное преобразование:
            none - без преобразования;
            elliptic - преобразование эллиптического параболоида;
            hyperbolic - преобразование гиперболического параболоида.
            
    Returns:
        testClasses: лист, содержащий датасет с классами неизвестных точек, определенных в алгоритме.
    """
    @staticmethod
    def classifyKNNCircle(trainingDots, unknownDots, k, kernelFunction, metrics, coordinateTransformation):
        training = trainingDots
        testClasses = []
        for i in range(len(unknownDots)):
            dot_class = DatasetProcessing.classifyDotCircle(training, unknownDots[i][0], k, kernelFunction, metrics,
                                                            coordinateTransformation)
            training.append(unknownDots[i])
            testClasses.append(dot_class)
        return testClasses
