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

    """Вычисление метрики полноты Recall (чувствительности Sensitivity).
    
    Recall = Sensitivity = TRP = TP / TP + FN = TP / P.
    Recall - полнота: доля найденных алгоритмом точек, принадлежащих  классу, относительно всех точек
     этого класса в тестовой выборке.
    
    Args:
        algorithmDotClasses: лист, содержащий датасет с классами неизвестных точек, определенных в алгоритме.
        startDatasetDotClasses: лист, содержащий датасет с классами неизвестных точек, определенных в начальном
         датасете.
    
    Returns:
        TRP = Recall.
    """
    @staticmethod
    def computingRecall(algorithmDotClasses, startDatasetDotClasses):
        TP, FP, TN, FN = Statistic.compareClasses(algorithmDotClasses, startDatasetDotClasses)
        P = TP + FN
        TPR = TP / P
        print('Recall: ' + str(TPR))
        return TPR

    """Вычисление метрики специфичности Specificity.
    Specificity = SPC = TN / FP + TN = TN / N.
    Specificity - специфичность: доля истинно отрицательных точек, найденных алгоритмом
     в тестовой выборке.

    Args:
        algorithmDotClasses: лист, содержащий датасет с классами неизвестных точек, определенных в алгоритме.
        startDatasetDotClasses: лист, содержащий датасет с классами неизвестных точек, определенных в начальном
         датасете.

    Returns:
        SPC = Specificity.
    """
    @staticmethod
    def computingSpecificity(algorithmDotClasses, startDatasetDotClasses):
        TP, FP, TN, FN = Statistic.compareClasses(algorithmDotClasses, startDatasetDotClasses)
        N = FP + TN
        SPC = TN / N
        print('Specificity: ' + str(SPC))
        return SPC

    """Вычисление метрики точности Precision.
    Precision = PPV = TP / (TP + FP).
    Precision - точность: доля точек, действительно принадлежащих данному классу, относительно всех точек,
     которые алгоритм отнес к этому классу.

    Args:
        algorithmDotClasses: лист, содержащий датасет с классами неизвестных точек, определенных в алгоритме.
        startDatasetDotClasses: лист, содержащий датасет с классами неизвестных точек, определенных в начальном
         датасете.

    Returns:
        PPV = Precision.
    """
    @staticmethod
    def computingPrecision(algorithmDotClasses, startDatasetDotClasses):
        TP, FP, TN, FN = Statistic.compareClasses(algorithmDotClasses, startDatasetDotClasses)
        PPV = TP / (TP + FP)
        print('Precision: ' + str(PPV))
        return PPV

    """Вычисление метрики правильности Accuracy.
    Accuracy = ACC = (TP + TN) / (TP + FN + FP + TN) = (TP + TN) / (P + N).
    Accuracy - правильность: доля точек, верно найденных алгоритмом.

    Args:
        algorithmDotClasses: лист, содержащий датасет с классами неизвестных точек, определенных в алгоритме.
        startDatasetDotClasses: лист, содержащий датасет с классами неизвестных точек, определенных в начальном
         датасете.

    Returns:
        ACC = Accuracy.
    """
    @staticmethod
    def computingAccuracy(algorithmDotClasses, startDatasetDotClasses):
        TP, FP, TN, FN = Statistic.compareClasses(algorithmDotClasses, startDatasetDotClasses)
        P = TP + FN
        N = FP + TN
        ACC = (TP + TN) / (P + N)
        print('Accuracy: ' + str(ACC))
        return ACC

    """Вычисление F1-меры для конечной оценки алгоритма.
    
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
    def computingF1_measure(algorithmDotClasses, startDatasetDotClasses):
        TPR = Statistic.computingRecall(algorithmDotClasses, startDatasetDotClasses)
        PPV = Statistic.computingPrecision(algorithmDotClasses, startDatasetDotClasses)
        F1_measure = 2 * TPR * PPV / (TPR + PPV)
        print('F1_measure: ' + str(F1_measure))
        return 0
