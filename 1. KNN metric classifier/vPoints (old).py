import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.lines
import matplotlib.path
import random
import math
import numpy as np


'''
Обработка входного датасета и подсчет количества точек
'''
def getDataset():
    # # метод с хабра # data = []

    linesCount = 0  # общее количество точек в датасете

    # переменные для разбиения строки датасета на значения
    dot_x = []
    dot_y = []
    dot_class = []

    # массивы для хранения обучающих координат
    training0_x = []
    training0_y = []
    training1_x = []
    training1_y = []

    # массивы для хранения тестовых координат
    test0_x = []
    test0_y = []
    test1_x = []
    test1_y = []

    # генерация количество обучающих значений
    trainingNumberGroup0 = random.randint(1, 59)
    trainingNumberGroup1 = random.randint(1, 59)

    # вычисление количества тренировочных значений
    testNumberGroup0 = 59 - trainingNumberGroup0
    testNumberGroup1 = 59 - trainingNumberGroup1

    for line in open('dataset.txt').readlines():
        # разделение входного датасета и заполение списка data
        newline = ""
        newline = line.replace(',', ' ')
        dot_x, dot_y, dot_class = newline.split()

    # # метод с хабра # data.append([[float(dot_x), float(dot_y)], int(dot_class)])
        # запись обучающей выборки
        if ((dot_class == '0') and (len(training0_x) <= trainingNumberGroup0)):
            training0_x.append([float(dot_x)])
            training0_y.append([float(dot_y)])
        if ((dot_class == '1') and (len(training1_x) <= trainingNumberGroup1)):
            training1_x.append([float(dot_x)])
            training1_y.append([float(dot_y)])

        # запись тестовой выборки
        if ((dot_class == '0') and (len(training0_x) > trainingNumberGroup0)):
            test0_x.append([float(dot_x)])
            test0_y.append([float(dot_y)])
        if ((dot_class == '1') and (len(training1_x) > trainingNumberGroup1)):
            test1_x.append([float(dot_x)])
            test1_y.append([float(dot_y)])

        # подсчет общего количества точек
        linesCount = linesCount + 1

    print("Total dots:", linesCount)
    print("Total dots from group 0:", 59)
    print("Total dots from group 1:", 59)

    print("Training dots from group 0:", trainingNumberGroup0)
    print("Training dots from group 1:", trainingNumberGroup1)

    print("Test dots from group 0:", testNumberGroup0)
    print("Test dots from group 1:", testNumberGroup1)

    # trainingData, testData
    return training0_x, training0_y, training1_x, training1_y, test0_x, test0_y, test1_x, test1_y


'''
Отображение групп точек на графике
'''
def buildPlot(training0_x, training0_y, training1_x, training1_y, test0_x, test0_y, test1_x, test1_y):
    # # метод с хабра
    # # colorMap = ListedColormap(['r', 'g'])
    # # plt.scatter([data[i][0][0] for i in range(len(data))],
    # #           [data[i][0][1] for i in range(len(data))],
    # #           c=[data[i][1] for i in range(len(data))],
    # #           cmap=colorMap)

    # 0tr 0te 1tr 1te
    colors = ['#FF0000', '#00CC00', '#FF7400', '#009999']
    groupTraining0 = plt.scatter(training0_x, training0_y, color=colors[0])
    groupTraining1 = plt.scatter(training1_x, training1_y, color=colors[1])
    groupTest0 = plt.scatter(test0_x, test0_y, color=colors[2], alpha=0.15)
    groupTest1 = plt.scatter(test1_x, test1_y, color=colors[3], alpha=0.15)
    #groupUnknown = plt.scatter(test0_x[0], test0_y[0], colors=colors[3])

    plt.title('Отображение входного датасета')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend((groupTraining0, groupTraining1, groupTest0, groupTest1),
               ('training-0', 'training-1', 'test-0', 'test-1'),
               loc='upper right')
    plt.show()
    return 0


'''
Отображение обучающей последовательности и неизвестной точки
вычисление центроидного классификатора
'''
def buildPlotWithUnknownDot(training0_x, training0_y, training1_x, training1_y, test0_x, test0_y, test1_x, test1_y):
    # 0tr 0te 1tr 1te
    colors = ['#FF0000', '#00CC00', '#FF7400', '#009999']
    groupTraining0 = plt.scatter(training0_x, training0_y, color=colors[0])
    groupTraining1 = plt.scatter(training1_x, training1_y, color=colors[1])
    # groupTest0 = plt.scatter(test0_x, test0_y, color=colors[2], alpha=0.15)
    # groupTest1 = plt.scatter(test1_x, test1_y, color=colors[3], alpha=0.15)
    unknownDot = plt.scatter(test0_x[1], test0_y[1], color=colors[3])

    # вывод координат неизвестной точки
    unknown_x = 0.0
    unknown_y = 0.0
    for i in range(len(test0_x)):
        if (i == 0):
            unknown_x += sum(test0_x[1])
            unknown_y += sum(test0_y[1])
    print('кооридинаты неизвестной точки', unknown_x, unknown_y)

    # вычисление центроидного классификатора
    # группа 0
    sumTraining0_x = 0.0
    sumTraining0_y = 0.0
    for i in range(len(training0_x)):
        sumTraining0_x += sum(training0_x[i])
        sumTraining0_y += sum(training0_y[i])
    center0_x = sumTraining0_x/len(training0_x)
    center0_y = sumTraining0_y/len(training0_x)
    groupTrainingCenter0 = plt.scatter(center0_x, center0_y, marker='*', color=colors[0])
    print('координаты центра группы 0:', center0_x, center0_y)
    # группа 1
    sumTraining1_x = 0.0
    sumTraining1_y = 0.0
    for i in range(len(training1_x)):
        sumTraining1_x += sum(training1_x[i])
        sumTraining1_y += sum(training1_y[i])
    center1_x = sumTraining1_x / len(training1_x)
    center1_y = sumTraining1_y / len(training1_x)
    groupTrainingCenter1 = plt.scatter(center1_x, center1_y, marker='*', color=colors[1])
    print('координаты центра группы 1:', center1_x, center1_y)

    # вычисление расстояний от центров до неизвестной точки
    center0_xToUnknown = abs(unknown_x - center0_x)
    center0_yToUnknown = abs(unknown_y - center0_y)
    center1_xToUnknown = abs(unknown_x - center1_x)
    center1_yToUnknown = abs(unknown_y - center1_y)

    distance0 = math.sqrt(abs(unknown_x - center0_xToUnknown)*abs(unknown_x - center0_xToUnknown) +
                          abs(unknown_y - center0_yToUnknown)*abs(unknown_y - center0_yToUnknown))
    distance1 = math.sqrt(abs(unknown_x - center1_xToUnknown)*abs(unknown_x - center1_xToUnknown) +
                          abs(unknown_y - center1_yToUnknown)*abs(unknown_y - center1_yToUnknown))

    print('из центра 0:', distance0)
    print('из центра 1:', distance1)
    if (distance0 < distance1):
        print('неизвестная точка принадлежит группе 0')
    else:
        print('неизвестная точка принадлежит группе 1')
    plt.plot([unknown_x, center0_x], [unknown_y, center0_y], color=colors[3])
    plt.plot([unknown_x, center1_x], [unknown_y, center1_y], color=colors[3])
    plt.title('Отображение обучающей последовательности и неизвестной точки')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend((groupTraining0, groupTraining1, unknownDot, groupTrainingCenter0, groupTrainingCenter1),
               ('training-0', 'training-1', 'unknown', 'Center-0', 'Center-1'),
               loc='upper right')
    plt.show()
    return 0


'''
TODO поиск координат исходных точек (до вычисления расстояния) для отображения отрезка радиуса
(проблема: иногда радиус захватывает больше точек)
'''
def buildPlotCircle(training0_x, training0_y, training1_x, training1_y, test0_x, test0_y, test1_x, test1_y):
    # 0tr 0te 1tr 1te
    colors = ['#FF0000', '#00CC00', '#FF7400', '#009999']
    groupTraining0 = plt.scatter(training0_x, training0_y, color=colors[0])
    groupTraining1 = plt.scatter(training1_x, training1_y, color=colors[1])
    # groupTest0 = plt.scatter(test0_x, test0_y, color=colors[2], alpha=0.15)
    # groupTest1 = plt.scatter(test1_x, test1_y, color=colors[3], alpha=0.15)
    unknownDot = plt.scatter(test0_x[1], test0_y[1], color=colors[3])

    # вывод координат неизвестной точки
    unknown_x = 0.0
    unknown_y = 0.0
    for i in range(len(test0_x)):
        if (i == 0):
            unknown_x += sum(test0_x[1])
            unknown_y += sum(test0_y[1])
    print('кооридинаты неизвестной точки', unknown_x, unknown_y)

    # вычисление k ближайших соседей
    # группа 0,1
    dTraining0_x = 0.0
    dTraining0_y = 0.0
    dTraining1_x = 0.0
    dTraining1_y = 0.0
    arrayDistance = []
    for i in range(len(training0_x)):
        dTraining0_x += sum(training0_x[i])
        dTraining0_y += sum(training0_y[i])
        distance0 = math.sqrt(abs(unknown_x - dTraining0_x)*abs(unknown_x - dTraining0_x) +
                              abs(unknown_y - dTraining0_y)*abs(unknown_y - dTraining0_y))
        print('distance0=', distance0, 'i=', i, 'dTraining0_x=', dTraining0_x, 'dTraining0_y=', dTraining0_y)
        dTraining0_x = 0.0
        dTraining0_y = 0.0
        arrayDistance.append([[float(distance0)], int(0), int(i)])

    for i in range(len(training1_x)):
        dTraining1_x += sum(training1_x[i])
        dTraining1_y += sum(training1_y[i])
        distance1 = math.sqrt(abs(unknown_x - dTraining1_x)*abs(unknown_x - dTraining1_x) +
                              abs(unknown_y - dTraining1_y)*abs(unknown_y - dTraining1_y))
        print('distance1=', distance1, 'i=', i, 'dTraining0_x=', dTraining1_x, 'dTraining0_y=', dTraining1_y)
        dTraining1_x = 0.0
        dTraining1_y = 0.0
        arrayDistance.append([[float(distance1)], int(1), int(i)])

    print('список до сортировки:', arrayDistance)
    # сортировка
    n = 1
    while n < len(arrayDistance):
        for i in range(len(arrayDistance) - n):
            if (arrayDistance[i] > arrayDistance[i+1]):
                arrayDistance[i], arrayDistance[i+1] = arrayDistance[i+1], arrayDistance[i]
        n += 1
    print('список после сортировки:', arrayDistance)

    k = 6
    #
    print('радиус окружности:', float(arrayDistance[k - 1][1]))
    testDistance = arrayDistance[k - 1][0]
    print('радиус окружности:', arrayDistance[k - 1])
    print('testDistance:', testDistance)
    print('type testDistance:', type(testDistance))
    print('float(testDistance[0]):', float(testDistance[0]))
    #
    # одинаковый размер осей
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    # circle = plt.Circle((unknown_x, unknown_y), float(testDistance[0]), fill=False)
    # plt.gcf().gca().add_artist(circle)
    circle = matplotlib.patches.Circle((unknown_x, unknown_y), float(testDistance[0]), fill=False)
    plt.gca().add_patch(circle)
    # plt.plot([unknown_x, center1_x], [unknown_y, center1_y], color=colors[3])
    plt.title('k ближайших соседей')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend((groupTraining0, groupTraining1, unknownDot),
               ('training-0', 'training-1', 'unknown'),
               loc='lower left')
    plt.show()
    return 0

if __name__ == '__main__':
    training0_x, training0_y, training1_x, training1_y, test0_x, test0_y, test1_x, test1_y = getDataset()
    #buildPlot(training0_x, training0_y, training1_x, training1_y, test0_x, test0_y, test1_x, test1_y)
    buildPlotWithUnknownDot(training0_x, training0_y, training1_x, training1_y, test0_x, test0_y, test1_x, test1_y)
    buildPlotCircle(training0_x, training0_y, training1_x, training1_y, test0_x, test0_y, test1_x, test1_y)