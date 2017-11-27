def get_metrics(bayes, data, debug):
    # true positive is a spam which is predicted well
    # true negative is a not spam which is predicted well
    # false positive is a spam which is not predicted well
    # false negative is a not spam which is not predicted well
    # [[ true positive, true negative ], [ false positive, false negative ]]
    matrix = [[0, 0], [0, 0]]

    for doc in data:
        if doc.is_spam:
            if bayes.classify(doc):
                matrix[0][0] += 1
                type_string = "spam"
            else:
                matrix[1][0] += 1
                type_string = "legit"
        else:
            if bayes.classify(doc):
                matrix[1][1] += 1
                type_string = "ham"
            else:
                matrix[0][1] += 1
                type_string = "legit"

        if debug:
            print("{} classified as {}".format(doc.name, type_string))

    if matrix[0][0] != 0:
        recall = matrix[0][0] / (matrix[0][0] + matrix[1][1])
        precision = matrix[0][0] / (matrix[0][0] + matrix[1][0])
        f_measure = 2 * precision * recall / (precision + recall)
    else:
        f_measure = 0

    return f_measure, matrix


class Validator(object):
    def __init__(self):
        pass

    @staticmethod
    def validate(bayes, data, debug=False, cross_valid=False):

        if not cross_valid:
            train_set = sum(data[:-5], [])
            test_set = data[-5]
            bayes.train(train_set)
            return get_metrics(bayes, test_set, debug)

        else:
            f_measures = []
            matrix_full = [[0, 0], [0, 0]]

            for i in range(len(data)):
                first_part = data[:(len(data) - 1) - i]
                second_part = data[(len(data) - 1) - i + 1:]

                train_set = first_part + second_part
                train_set = sum(train_set, [])
                test_set = data[(len(data) - 1) - i]

                bayes.reset()
                bayes.train(train_set)

                f_measure, matrix = get_metrics(bayes, test_set, debug)

                matrix_full[0][0] += matrix[0][0]
                matrix_full[0][1] += matrix[0][1]
                matrix_full[1][0] += matrix[1][0]
                matrix_full[1][1] += matrix[1][1]
                f_measures.append(f_measure)

            matrix_full[0][0] = round(matrix_full[0][0] / len(f_measures))
            matrix_full[0][1] = round(matrix_full[0][1] / len(f_measures))
            matrix_full[1][0] = round(matrix_full[1][0] / len(f_measures))
            matrix_full[1][1] = round(matrix_full[1][1] / len(f_measures))

            return sum(f_measures) / len(f_measures), matrix_full
