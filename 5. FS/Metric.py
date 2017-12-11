from Pearson import Pearson
from Spearman import Spearman
from IG import IG


class Metric(object):
    def __init__(self, name):
        if name == 'pearson':
            self.metric = Pearson
        elif name == 'spearman':
            self.metric = Spearman
        elif name == 'ig':
            self.metric = IG

    def build(self, X, Y, logs=False, limit=100):
        metric = self.metric(X, Y, logs)
        metric.fit()
        metric.process()
        return metric.getKeys()[:limit]
