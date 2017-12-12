# Linear regression
[Table of contents](https://github.com/fedy95/MachineLearning/blob/master/README.md)

## Used software:
- Python-IDE: PyCharm Community 2017.2;
- Project Interpreter: python 3.5.3 (amd64);
- Used python packeges:
	- matplotlib v2.1.0;
	- tabulate v0.8.1.*

### Problem
Требуется попробовать устроить FS на предоставленном наборе данных с помощью фильтров.
Можно использовать любые три метрики, какие найдете (например Spearman, Pearson, IG).
Попробовать как-то разумно выбрать сколько признаков оставлять. Сравнить эффективность
работы классификатора на полученных фичах (например SVM). Хотелось бы увидеть красивую
картинку какие фичи куда попали, а куда не попали (например что-то похожее на круги Эйлера
или диаграмму совпадений).

### Program structure

### Output
```
Comparable Table
| Kernel   | Metric   |   Filter limit |   SVM C |   F-mature | Confusion Matrix     |
|----------+----------+----------------+---------+------------+----------------------|
| linear   | pearson  |           1000 |       1 |   0.548148 | [[37, 2], [54, 7]]   |
| linear   | spearman |           1000 |       1 |   0.615385 | [[28, 37], [19, 16]] |
| linear   | ig       |           1000 |       1 |   0.517857 | [[29, 17], [39, 15]] |
```

### FAQ

### Links
1) Коэффициент корреляции Спирмена: http://www.machinelearning.ru/wiki/index.php?title=%D0%9A%D0%BE%D1%8D%D1%84%D1%84%D0%B8%D1%86%D0%B8%D0%B5%D0%BD%D1%82_%D0%BA%D0%BE%D1%80%D1%80%D0%B5%D0%BB%D1%8F%D1%86%D0%B8%D0%B8_%D0%A1%D0%BF%D0%B8%D1%80%D0%BC%D0%B5%D0%BD%D0%B0
2) Коэффициент корреляции Пирсона: http://www.machinelearning.ru/wiki/index.php?title=%D0%9A%D0%BE%D1%8D%D1%84%D1%84%D0%B8%D1%86%D0%B8%D0%B5%D0%BD%D1%82_%D0%BA%D0%BE%D1%80%D1%80%D0%B5%D0%BB%D1%8F%D1%86%D0%B8%D0%B8_%D0%9F%D0%B8%D1%80%D1%81%D0%BE%D0%BD%D0%B0
