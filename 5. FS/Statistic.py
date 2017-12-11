class Statistic(object):
    def __init__(self):
        pass

    @staticmethod
    def get_statistics(logs, pearson_keys, spearman_keys, ig_keys):
        All = []
        pearson = []
        pearson_spearman = []
        pearson_IG = []
        spearman = []
        spearman_IG = []
        IG = []

        for x in pearson_keys:
            if x in spearman_keys and x in ig_keys:
                All.append(x)
            elif x in spearman_keys:
                pearson_spearman.append(x)
            elif x in ig_keys:
                pearson_IG.append(x)
            else:
                pearson.append(x)

        for x in spearman_keys:
            if x in pearson_keys and x in ig_keys and x not in All:
                All.append(x)
            elif x in pearson_keys and x not in pearson_spearman:
                pearson_spearman.append(x)
            elif x in ig_keys:
                spearman_IG.append(x)
            else:
                spearman.append(x)

        for x in ig_keys:
            if x in pearson_keys and x in spearman_keys and x not in All:
                All.append(x)
            elif x in pearson_keys and x not in pearson_IG:
                pearson_IG.append(x)
            elif x in spearman_keys and x not in spearman_IG:
                spearman_IG.append(x)
            else:
                IG.append(x)

        if logs:
            print("\n")
            print("All: ", All)
            print('pearson_spearman: ', pearson_spearman)
            print('pearson_IG: ', pearson_IG)
            print('spearman_IG: ', spearman_IG)
            print('pearson: ', pearson)
            print('spearman: ', spearman)
            print('IG: ', IG)

        return All, pearson_spearman, pearson_IG, spearman_IG, pearson, spearman, IG

    ##
    # Get F-measure by Confusion Matrix
    # @matrix => [[Number, Number], [Number, Number]]
    ##
    @staticmethod
    def get_f_measure(matrix):
        if matrix[0][0] != 0:
            recall = float(matrix[0][0]) / (float(matrix[0][0]) + float(matrix[1][1]))
            precision = float(matrix[0][0]) / (float(matrix[0][0]) + float(matrix[1][0]))
            return 2 * precision * recall / (precision + recall)
        else:
            return 0

    ##
    # Get Metrics By Array
    # @predict  => [Number, Number]
    # @start    => [Number, Number]
    ##
    @staticmethod
    def get_metrics(predict, start):
        # true positive is a spam which is predicted well
        # true negative is a not spam which is predicted well
        # false positive is a spam which is not predicted well
        # false negative is a not spam which is not predicted well
        # [[ true positive, true negative ], [ false positive, false negative ]]
        matrix = [[0, 0], [0, 0]]
        for i in range(len(predict)):
            if int(predict[i]) == int(start[i]):
                if int(predict[i]) == 1:
                    matrix[0][0] += 1
                else:
                    matrix[0][1] += 1
            else:
                if int(predict[i]) == 1:
                    matrix[1][0] += 1
                else:
                    matrix[1][1] += 1
        return matrix
