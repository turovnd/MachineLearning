from Documents import Documents
from BayesClassifier import Bayes
from CrossValidation import Validator
from tabulate import tabulate

if __name__ == "__main__":
    documents = Documents.get_all_docs('Bayes\pu1', 'spmsg')

    ##
    # For Bayer Classification
    # @task - ограничения на ваш классификатор, чтобы хорошие письма практически никогда не попадали в спам, но при этом, возможно, общее качество классификации несколько уменьшилось
    ##
    include_header = True       # True || False
    minimum_occurrence = 3      # None || Number        => при классификации участвуют слова, у которых минимальное кол-во раз появление слова в тренировочной выборке > minimum_occurrence
    discard_deviation = 0.3     # None || Number < 0.5  => отбросить отклонение
    top_n = None                # None || Number        => в классификации участвуют только `top_n` слов в документе, которые имеют наивысшие веса

    bayes = Bayes(include_header=include_header, minimum_occurrence=minimum_occurrence,
                  discard_deviation=discard_deviation, top_n=top_n)

    ##
    # For Cross Validation
    ##
    debug_print = False         # True || False
    cross_validation = True     # True || False

    f_measure, matrix = Validator.validate(bayes, documents, debug_print, cross_validation)

    table = [["T", matrix[0][0], matrix[0][1]], ["F", matrix[1][0], matrix[1][1]]]
    print(tabulate(table,
                   headers=["", "P", "N"],
                   tablefmt='orgtbl'))

    print("\nF-measure: ", f_measure)
