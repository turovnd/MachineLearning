import random
import math
import numpy as np
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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


"""Метод визуализации полученных датасетов на графике двумерного простраства (x,y).

График включает четыре группы точек: две обучающих и две тестирующих выборки.
Применяется после вызова метода getDataset(t).
Args:
    trainingDotsWithClass: лист, содержащий датасет обучающих точек в виде ([x,y],classDot).
    testDotsWithClass: лист, содержащий датасет тестирующих точек в виде ([x,y],classDot).

Returns:
    0: удачное исполнение.
"""
def buildPlot(trainingDotsWithClass, testDotsWithClass):
    trainingDots0 = getDotsByClass(trainingDotsWithClass, 0)
    trainingDots1 = getDotsByClass(trainingDotsWithClass, 1)
    testDots0 = getDotsByClass(testDotsWithClass, 0)
    testDots1 = getDotsByClass(testDotsWithClass, 1)
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


"""Метод формирования датасета точек одного класса в виде ([x,y]).

Args:
    DotsWithClass: лист, содержащий датасет точек в виде ([x,y],classDot).
    classDot: номер класса датасета точек.

Returns:
    dots: лист, содержащий датасет точек в виде ([x,y]).
"""
def getDotsByClass(DotsWithClass, classDot):
    dots = []
    for i in range(len(DotsWithClass)):
        if DotsWithClass[i][1] == classDot:
            dots.append(DotsWithClass[i][0])
    return dots


"""#TODO в отдельный файл (кусок не обязателен).Метод визуализации обучающего датасета и неизвестной точки
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
def buildPlotWithUnknownDot(trainingDotsWithClass, testDotsWithClass, num):
    colors = ['red', 'blue', 'green']

    if num > len(testDotsWithClass):
        print("error")
        return -1

    trainingDots0 = getDotsByClass(trainingDotsWithClass, 0)
    trainingDots1 = getDotsByClass(trainingDotsWithClass, 1)
    testDot = testDotsWithClass[num]
    groupTraining0 = plt.scatter([trainingDots0[i][0] for i in range(len(trainingDots0))],
                                 [trainingDots0[i][1] for i in range(len(trainingDots0))],
                                 color=colors[0])
    groupTraining1 = plt.scatter([trainingDots1[i][0] for i in range(len(trainingDots1))],
                                 [trainingDots1[i][1] for i in range(len(trainingDots1))],
                                 color=colors[1])
    centroid0 = getCentroid(trainingDots0)
    centroid1 = getCentroid(trainingDots1)
    groupTrainingCenter0 = plt.scatter(centroid0[0], centroid0[1], marker='*', color=colors[0])
    groupTrainingCenter1 = plt.scatter(centroid1[0], centroid1[1], marker='*', color=colors[1])
    unknownDot = plt.scatter(testDot[0][0], testDot[0][1], color=colors[2])
    euclideanDistance0 = getEuclideanDistance(testDot[0], centroid0)
    euclideanDistance1 = getEuclideanDistance(testDot[0], centroid1)

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


"""Метод визуализации обучающего датасета и неизвестной точки
 на графике двумерного простраства (x,y) c помощью вычисления наибольшего 
 количества ближайших соседей.

Максимальное количество точек одного класса в круге определяет класс тестируемой точки.

Args:
    trainingDotsWithClass: лист, содержащий датасет обучающих точек в виде ([x,y],classDot).
    testDotsWithClass: лист, содержащий датасет тестирующих точек в виде ([x,y],classDot).
    num: номер тестового значения (частный случай, в общем случае прогоняются все точки
     без графиков).
    k: количество соседей тестируемой точки, для определения её класса.
    
Returns:
    0: удачное исполнение.    
"""
def buildPlotCircle(trainingDotsWithClass, testDotsWithClass, num, k):
    colors = ['red', 'blue', 'green']

    if num > len(testDotsWithClass):
        print("error")
        return -1

    trainingDots0 = getDotsByClass(trainingDotsWithClass, 0)
    trainingDots1 = getDotsByClass(trainingDotsWithClass, 1)
    groupTraining0 = plt.scatter([trainingDots0[i][0] for i in range(len(trainingDots0))],
                                 [trainingDots0[i][1] for i in range(len(trainingDots0))],
                                 color=colors[0])
    groupTraining1 = plt.scatter([trainingDots1[i][0] for i in range(len(trainingDots1))],
                                 [trainingDots1[i][1] for i in range(len(trainingDots1))],
                                 color=colors[1])
    testDot = testDotsWithClass[num]
    unknownDot = plt.scatter(testDot[0][0], testDot[0][1], color=colors[2])
    testDistance = []

    for i in range(len(trainDots)):
        testDistance.append([getEuclideanDistance(testDot[0], trainDots[i][0]), trainDots[i][1]])
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

    # тестовые данные
    if class_0 > class_1:
        print('неизвестная точка принадлежит классу 0')
    else:
        print('неизвестная точка принадлежит классу 1')

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


"""Метод расчета евлидового расстояния для двух точек двумерного пространства (x,y).

Формула: p(x,y) = sqrt((x0[i]-x1[i])^2)+(y0[i]-y1[i])^2).

Args:
    trainingDot: совокупность координат (x,y) обучающей точки (из обучающей выборки).
    unknownDot:  совокупность координат (x,y) неизвестной точки (из тестирующей выборки).
    
Returns:
    Действительное число, евклидово расстояние между двумя точками.
"""
def getEuclideanDistance(trainingDot, unknownDot):
    return math.sqrt(math.pow(trainingDot[0] - unknownDot[0], 2)
                     + math.pow(trainingDot[1] - unknownDot[1], 2))


"""#TODO в отдельный файл (кусок не обязателен).Метод расчета центрода класса точек двумерного пространства (x,y).

Формула: sum(x[i]/len(x)).

Args:
    trainingDots: лист, содержащий датасет обучающих точек в виде ([x,y]).

Returns: 
    Лист, содержащий координату центра класса в виде (x,y).
"""
def getCentroid(trainingDots):
    x = sum([trainingDots[i][0] for i in range(len(trainingDots))])
    y = sum([trainingDots[i][1] for i in range(len(trainingDots))])
    return [x / len(trainingDots), y / len(trainingDots)]


"""#TODO в отдельный файл (кусок не обязателен).Метод определения класса тестируемой точки
 с помощью центроидов.

Args:
    trainingDot0: лист, содержащий датасет обучающих точек класса 0 в виде ([x,y]).
    trainingDot1: лист, содержащий датасет обучающих точек класса 1 в виде ([x,y]).
    unknownDot:  совокупность координат (x,y) неизвестной точки (из тестирующей выборки).
    
Returns:
    0: неизвестная точка принадлежит классу 0.    
    1: неизвестная точка принадлежит классу 1.    
"""
def classifyDotCentroid(trainingDot0, trainingDot1, unknownDot):
    centerDots0 = getCentroid(trainingDot0)
    centerDots1 = getCentroid(trainingDot1)
    euclideanDistance0 = getEuclideanDistance(unknownDot, centerDots0)
    euclideanDistance1 = getEuclideanDistance(unknownDot, centerDots1)
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
def classifyDotCircle(trainingDotsWithClass, testDotsWithClass, k, core):
    testDist = []
    for i in range(len(trainingDotsWithClass)):
        testDist.append([getEuclideanDistance(testDotsWithClass, trainingDotsWithClass[i][0]), trainingDotsWithClass[i][1]])

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
                class_1 = 1 / (math.sqrt(2*math.pi) * math.exp(-1/2*math.pow(class_1,2)))
            else:
                class_0 = 1 / (math.sqrt(2*math.pi) * math.exp(-1/2*math.pow(class_0,2)))
    elif core == "logistic":
        class_0 = 0
        class_1 = 0
        for i in range(k):
            if testDist[i][1] == 1:
                class_1 = 1 / (math.exp(class_1) + 2 + math.exp(-class_1))
            else:
                class_0 = 1 / (math.exp(class_0 ) + 2 + math.exp(-class_0 ))
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
def classifyKNNCentroid(trainingDots, unknownDots):
    trainingDots0 = getDotsByClass(trainingDots, 0)
    trainingDots1 = getDotsByClass(trainingDots, 1)
    testClasses = []
    for i in range(len(unknownDots)):
        dot_class = classifyDotCentroid(trainingDots0, trainingDots1, unknownDots[i][0])
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
def classifyKNNCircle(trainingDots, unknownDots, k, core):
    training = trainingDots
    testClasses = []
    for i in range(len(unknownDots)):
        dot_class = classifyDotCircle(training, unknownDots[i][0], k,core)
        training.append(unknownDots[i])
        testClasses.append(dot_class)
    return testClasses


"""Метод сравнения совпадения номеров классов точек из начального датасета, с получившимися.

Args:
    algorithmDotClasses: лист, содержащий датасет с классами неизвестных точк, определенных в алгоритме.
    startDatasetDotClasses: лист, содержащий датасет с классами неизвестных точк, определенных в начальном датасете.
    
Returns:
    TP - True Positive: количество истино-положительных решений.
    FP - False Positive: количество ложно-положительных решений.
    FN - False Negative: количество ложно-отрицательных решений.
    TN - True Negative: количество истино-отрицательных решений.
"""
def getClassify(algorithmDotClasses, startDatasetDotClasses):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(startDatasetDotClasses)):
        if algorithmDotClasses[i] == startDatasetDotClasses[i]:
            if algorithmDotClasses[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if algorithmDotClasses[i] == 1:
                FP += 1
            else:
                FN += 1
    return TP, FP, TN, FN


"""Вычисление F1-меры для конечной оценки алгоритма

P = TP + FN : number of positive examples.
N = FP + TN : number of negative examples.
Precision - точность: доля точек, действительно принадлежащих данному классу, относительно всех точек,
 которые алгоритм отнес к этому классу.
Recall - полнота: доля найденных алгоритмом точек, принадлежащих классу, относительно всех точек
 этого класса в тестовой выборке.

Args:
    algorithmDotClasses: лист, содержащий датасет с классами неизвестных точек, определенных в алгоритме.
    startDatasetDotClasses: лист, содержащий датасет с классами неизвестных точек, определенных в начальном датасете.
    
Returns:
    0: удачное исполнение.
"""
def F1_measure(classified, not_classified):
    TP, FP, TN, FN = getClassify(classified, not_classified)
    P = TP + FN
    N = FP + TN
    Recall = TP / P
    Precision = TP / (TP + FP)
    F1_measure = 2 * Recall * Precision / (Recall + Precision)

    print('Recall: ' + str(Recall))
    print('Precision: ' + str(Precision))
    print('F1_measure: ' + str(F1_measure))
    return 0


if __name__ == '__main__':
    t = 50
    data, trainDots, testDots = getDataset(t)
    buildPlot(trainDots, testDots)
    buildPlotWithUnknownDot(trainDots, testDots, 50)
    # start
    testClasses = [testDots[i][1] for i in range(len(testDots))]
    print('testClasses', testClasses)

    print("*" * 50)
    print("classifyKNN_center")
    testClassesClassified = classifyKNNCentroid(trainDots, testDots)
    F1_measure(testClassesClassified, testClasses)

    k = 5
    core = ["gaussian", "logistic"]
    print("*" * 50)
    buildPlotCircle(trainDots, testDots, 50, k)
    print("classifyKNN_circle  without core")
    testClassesClassified = classifyKNNCircle(trainDots, testDots, k, 0)
    F1_measure(testClassesClassified, testClasses)

    print("*" * 50)
    print("classifyKNN_circle  " + core[0])
    # finish
    testClassesClassified = classifyKNNCircle(trainDots, testDots, k, core[0])
    print('testClassesClassified', testClassesClassified)
    F1_measure(testClassesClassified, testClasses)

    print("*" * 50)
    print("classifyKNN_circle  " + core[1])
    testClassesClassified = classifyKNNCircle(trainDots, testDots, k, core[1])
    F1_measure(testClassesClassified, testClasses)

    arr = []
    for i in range(len(testDots)):
        arr.append([[trainDots[i][0][0] * math.pi, trainDots[i][0][1]], trainDots[i][1]])
    trainDots = arr

    print("*" * 50)
    print("classifyKNN_circle  " + core[1] + "  пространственное преобразование")
    testClassesClassified = classifyKNNCircle(trainDots, testDots, k, core[1])
    F1_measure(testClassesClassified, testClasses)

    arr = []
    for i in range(len(testDots)):
        arr.append([[trainDots[i][0][0] * 10 * math.pi, trainDots[i][0][1] * 10 * math.pi], trainDots[i][1]])
    trainDots = arr

    print("*" * 50)
    print("classifyKNN_circle  " + core[1] + "  пространственное преобразование 2 ")
    testClassesClassified = classifyKNNCircle(trainDots, testDots, k, core[1])
    F1_measure(testClassesClassified, testClasses)
