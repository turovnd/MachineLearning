from matplotlib import pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from Statistic import Statistic


class Plot(object):
    def __init__(self):
        pass

    @staticmethod
    def euler(pearson_keys, spearman_keys, ig_keys):

        All, pearson_spearman, pearson_IG, spearman_IG, pearson, spearman, IG = Statistic.get_statistics(pearson_keys, spearman_keys, ig_keys)

        plt.figure(figsize=(30, 30))
        v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels=('Pearson', 'Spearman', 'IG'))

        c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
        v.get_label_by_id('100').set_text(str(pearson))
        v.get_label_by_id('010').set_text('\n\n' + str(spearman))
        v.get_label_by_id('001').set_text(str(IG))
        v.get_label_by_id('110').set_text(str(pearson_spearman))
        v.get_label_by_id('011').set_text('' + str(spearman_IG))
        v.get_label_by_id('101').set_text('\n\n' + str(pearson_IG))
        v.get_label_by_id('111').set_text(str(All))
        plt.title("Feature Selection")

        plt.show()
