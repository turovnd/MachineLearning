from DatasetProcessing import *

"""
"""


class Statistic(object):
    """Initialization variables"""
    def __init__(self):
        pass

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
    @staticmethod
    def compareClasses(algorithmDotClasses, startDatasetDotClasses):
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
    @staticmethod
    def computingF1_measure(classified, not_classified):
        TP, FP, TN, FN = Statistic.compareClasses(classified, not_classified)
        P = TP + FN
        N = FP + TN
        Recall = TP / P
        Precision = TP / (TP + FP)
        F1_measure = 2 * Recall * Precision / (Recall + Precision)

        print('Recall: ' + str(Recall))
        print('Precision: ' + str(Precision))
        print('F1_measure: ' + str(F1_measure))
        return 0
