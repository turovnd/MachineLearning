# Linear regression
[Table of contents](https://github.com/fedy95/MachineLearning/blob/master/README.md)

## Used software:
- Python-IDE: PyCharm Community 2017.2;
- Project Interpreter: python 3.5.3 (amd64);
- Used python packeges:
	- matplotlib v2.1.0;
	- tabulate v0.8.1.*

### Problem
Задача состоит в классификации спама. Спам сообщения содержат в своем названии spmsg, нормальные сообщения содержат legit.
Сам текст письма состоит из двух частей: темы и тела письма.
Все слова заменены на инты, соответствующие их индексу в некотором глобальном словаре (своего рода анонимизация).
Соответственно от вас требуется построить наивный Байесовский классификатор и при этом:

1) Придумать, либо протестировать, что можно делать с темой и телом письма для улучшения качества работы
2) Как учитывать (или не учитывать) слова, которые могут встретиться в обучающей выборке, но могут не встретится в тестовой или наоборот.
3) Как наложить дополнительные ограничения на ваш классификатор так, чтобы хорошие письма практически никогда не попадали в спам, но при этом, возможно, общее качество классификации несколько уменьшилось.
4) Понимать как устроен классификатор внутри и уметь отвечать на какие-никакие вопросы по теории с ним связанной.

Для написания классификатора не разрешается использовать библиотеки, наподобие weka и sklearn, а также реализации из них. Кросс-валидацию можно производить любыми средствами.

![datasets text files](https://github.com/fedy95/MachineLearning/tree/master/3.%20Naive%20Bayes%20classifier/Bayes/pu1)

### Program structure
- Documents.py
	- class Document    => return string `Document [name={}, header={}, content={}, is_spam={}`
	- class Documents:
	    - @staticmethod get_all_docs()  => return array of `Document`

- BayesClassifier.py
    - class Bayes_Classifier:
        - reset()     => set default counters of data, spam, ham
        - add_coeff() => increase counters of data, spam, ham
        - train()     => training model on `train_set`
        - classify()  => do classify of `document`

- CrossValidation.py
    - get_metrics()   => return F-measure and confusion matrix
    - class Validator:
        - @staticmethod validate() => validate documents

### Output

#### Наилучший результат с крос валидацией
```
|    |   P |   N |
|----+-----+-----|
| T  |  47 |  60 |
| F  |   1 |   1 |

F-measure:  0.9789395840099022
```

#### Наилучший результат без крос валидации

Половина докуметов - обучаущая выборка, оствшаяся - тестовая
```
|    |   P |   N |
|----+-----+-----|
| T  |  44 |  60 |
| F  |   4 |   1 |

F-measure:  0.946236559139785
```


### FAQ

Апостериорная вероятность — условная вероятность случайного события при условии того, что известны данные полученные после опыта.

### Links
1) http://datareview.info/article/6-prostyih-shagov-dlya-osvoeniya-naivnogo-bayesovskogo-algoritma-s-primerom-koda-na-python/
2) https://www.coursera.org/learn/supervised-learning/lecture/8RMf8/spam-fil-try-i-naivnyi-baiiesovskii-klassifikator
3) http://bazhenov.me/blog/2012/06/11/naive-bayes.html
