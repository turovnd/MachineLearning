from DatasetProcessing import *

"""

"""
class Statistic(object):

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
        TP, FP, TN, FN = DatasetProcessing.getClassify(classified, not_classified)
        P = TP + FN
        N = FP + TN
        Recall = TP / P
        Precision = TP / (TP + FP)
        F1_measure = 2 * Recall * Precision / (Recall + Precision)

        print('Recall: ' + str(Recall))
        print('Precision: ' + str(Precision))
        print('F1_measure: ' + str(F1_measure))
        return 0