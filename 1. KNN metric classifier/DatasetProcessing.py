import math


"""
"""


class DatasetProcessing(object):
    """Initialization variables"""
    def __init__(self):
        pass

    """Метод обработки входного датасета и подсчета количества точек.

    file 'chips.txt': входной датасет, содержащий информацию о точках в виде (x,y,classDot).

    Args:
        t: число, обозначающее количество точек, разделяющих входной датасет
         на обучающий, тестирующий датасет.
         (например, 10 первых с начала и конца входят в обучающий датасет,
         остальные в тестирующий).

    Returns:
        data: лист, содержащий входной датасет в виде ([x,y],classDot).
        trainingDotsWithClass: лист, содержащий датасет обучающих точек в виде ([x,y],classDot).
        testDotsWithClass: лист, содержащий датасет тестирующих точек в виде ([x,y],classDot).
    """
    @staticmethod
    def getDataset(t):
        data = []
        trainingDotsWithClass = []
        testDotsWithClass = []

        f = open('dataset.txt')
        for line in f:
            dot_x, dot_y, dot_class = line.split(',')
            data.append([[float(dot_x), float(dot_y)], int(dot_class)])
        f.close()
        for i in range(len(data)):
            if t % 2 == 0:
                if i < int(t / 2) or i >= len(data) - int(t / 2):
                    trainingDotsWithClass.append(data[i])
                else:
                    testDotsWithClass.append(data[i])
            else:
                if i <= int(t / 2) or i >= len(data) - int(t / 2):
                    trainingDotsWithClass.append(data[i])
                else:
                    testDotsWithClass.append(data[i])
        # тестовые данные
        print("Total dots:", len(data))
        print("Training dots:", len(trainingDotsWithClass))
        print("Test dots:", len(testDotsWithClass))
        return data, trainingDotsWithClass, testDotsWithClass

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

    """Метод расчета евлидового расстояния для двух точек двумерного пространства (x,y).
    Формула: p(x,y) = sqrt((x0[i]-x1[i])^2)+(y0[i]-y1[i])^2).
    
    Args:
        trainingDot: совокупность координат (x,y) обучающей точки (из обучающей выборки).
        unknownDot:  совокупность координат (x,y) неизвестной точки (из тестирующей выборки).
    
    Returns:
        Действительное число, евклидово расстояние между двумя точками.
    """
    @staticmethod
    def getEuclideanDistance(trainingDot, unknownDot):
        return math.sqrt(math.pow(trainingDot[0] - unknownDot[0], 2)
                         + math.pow(trainingDot[1] - unknownDot[1], 2))

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
        euclideanDistance0 = DatasetProcessing.getEuclideanDistance(unknownDot, centerDots0)
        euclideanDistance1 = DatasetProcessing.getEuclideanDistance(unknownDot, centerDots1)
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
        core: ядро используемой функции.
    
    Returns:
        0: неизвестная точка принадлежит классу 0.    
        1: неизвестная точка принадлежит классу 1.
    """
    @staticmethod
    def classifyDotCircle(trainingDotsWithClass, testDotsWithClass, k, core):
        testDist = []
        for i in range(len(trainingDotsWithClass)):
            testDist.append(
                [DatasetProcessing.getEuclideanDistance(testDotsWithClass, trainingDotsWithClass[i][0]),
                 trainingDotsWithClass[i][1]])

        n = 1
        while n < len(testDist):
            for i in range(len(testDist) - n):
                if testDist[i][0] > testDist[i + 1][0]:
                    testDist[i], testDist[i + 1] = testDist[i + 1], testDist[i]
            n += 1

        if core == "gaussian":
            class_0 = 0
            class_1 = 0
            for i in range(k):
                if testDist[i][1] == 1:
                    class_1 = 1 / (math.sqrt(2 * math.pi) * math.exp(-1 / 2 * math.pow(class_1, 2)))
                else:
                    class_0 = 1 / (math.sqrt(2 * math.pi) * math.exp(-1 / 2 * math.pow(class_0, 2)))
        elif core == "logistic":
            class_0 = 0
            class_1 = 0
            for i in range(k):
                if testDist[i][1] == 1:
                    class_1 = 1 / (math.exp(class_1) + 2 + math.exp(-class_1))
                else:
                    class_0 = 1 / (math.exp(class_0) + 2 + math.exp(-class_0))
        else:
            class_0 = 0
            class_1 = 0
            for i in range(k):
                if testDist[i][1] == 1:
                    class_1 += 1
                else:
                    class_0 += 1

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
        core: ядро используемой функции.
    
    Returns:
        testClasses: лист, содержащий датасет с классами неизвестных точек, определенных в алгоритме.
    """
    @staticmethod
    def classifyKNNCircle(trainingDots, unknownDots, k, core):
        training = trainingDots
        testClasses = []
        for i in range(len(unknownDots)):
            dot_class = DatasetProcessing.classifyDotCircle(training, unknownDots[i][0], k, core)
            training.append(unknownDots[i])
            testClasses.append(dot_class)
        return testClasses
