from collections import defaultdict
from math import log, e


class Bayes(object):
    def __init__(self, minimum_occurrence=5, discard_deviation=0.1, include_header=False, top_n=None):
        self.SPAM = 'spam'
        self.HAM = 'ham'

        self.minimum_occurrence = minimum_occurrence
        self.discard_deviation = discard_deviation
        self.include_header = include_header
        self.top_n = top_n

        self.data = {}
        self.all_ct = {}

        self.reset()

    def reset(self):
        self.data = defaultdict(lambda: defaultdict(lambda: 0.000001))
        self.all_ct = {self.SPAM: 0, self.HAM: 0}

    def add_coeff(self, words, is_spam):
            for word in words:
                if is_spam:
                    self.data[word][self.SPAM] += 1
                    self.all_ct[self.SPAM] += 1
                else:
                    self.data[word][self.HAM] += 1
                    self.all_ct[self.HAM] += 1

    # Train model
    def train(self, train_set):
        for document in train_set:
            self.add_coeff(document.content, document.is_spam)

            if self.include_header:
                self.add_coeff(document.header, document.is_spam)

    # Classify document
    def classify(self, document):
        def spam_prob(word):
            return self.data[word][self.SPAM] / (self.data[word][self.SPAM] + self.data[word][self.HAM])

        words = document.content

        if self.include_header:
            words += document.header

        # минимальное кол-во раз появление слова в тренировочной выборке
        if self.minimum_occurrence is not None:
            words = list(filter(lambda word: self.data[word][self.SPAM] + self.data[word][self.HAM] >= self.minimum_occurrence, words))

        # отбросить отклонение
        if self.discard_deviation is not None:
            words = list(filter(lambda word: abs(0.5 - spam_prob(word)) >= self.discard_deviation, words))

        # в классификации участвуют только `top_n` слов в документе, которые имеют наивысшие веса
        if self.top_n is not None:
            words.sort(key=lambda word: abs(0.5 - spam_prob(word)), reverse=True)
            words = words[:self.top_n]

        spam_probability = sum(map(lambda word: log(self.data[word][self.SPAM] / (self.all_ct[self.SPAM])), words))
        ham_probability = sum(map(lambda word: log(self.data[word][self.HAM] / (self.all_ct[self.HAM])), words))

        return spam_probability > ham_probability
