# Linear regression
[Table of contents](https://github.com/fedy95/MachineLearning/blob/master/README.md)

## Used software:
- Python-IDE: PyCharm Community 2017.2;
- Project Interpreter: python 2.7.* (amd64);
- Used python packeges:
    - numpy
    - random
    - tabulate
    - scipy.stats
    - matplotlib.patches
    - pylab
    - cvxopt

### Problem
Требуется написать SVM без библиотек, посчитать у него (разумеется) f-меру и confusion матрицу.
Далее требуется разобраться в том, что такое критерий Вилкоксона, и сравнить с его помощью работу
алгоритмов kNN и свежепосчитанный SVM, посчитать p-value. Всё это на старом наборе данных chips.txt
из первой лабы. Уметь отвечать на любую теорию, как про SVM, так и про статистический тест.

### Program structure
- Dataset
    - updateTrainTest
    - getDotsByMode
- Kernel
    - linear_kernel
    - polynomial_kernel
    - gaussian_kernel
    - get
- Metric
    - euclidean
    - manhattan
    - get
- Plot
    - smv
    - knn
- CrossValidation
    - get_f_measure
    - get_metrics
    - Validator
        - svm_validate
        - knn_validate
- KNN
    - fit
    - predict
    - classify
- SVM
    - fit
    - predict
    - project
- main - start point

### Output
Best result on
```
kernel      =>  'polynomial'
metric      =>  'manhattan'
svm_C       =>  1000
k_neighbors =>  5

SVM
|    |   P |   N |
|----+-----+-----|
| T  |  50 |  48 |
| F  |  12 |   8 |
F-measure: 0.833333333333

kNN
|    |   P |   N |
|----+-----+-----|
| T  |  45 |  47 |
| F  |  13 |  13 |
F-measure: 0.775862068966

P-value: 0.285049407403
```

### FAQ

### Links
1) Course of creating custom CSV: https://youtu.be/AbVtcUBlBok + https://youtu.be/QAs2olt7pJ4 + https://youtu.be/VhHLpg7ZS4Q + https://youtu.be/yrnhziJk-z8
2) SVM algorithm: https://gist.github.com/mblondel/586753
3) Классификация данных методом SVM: https://habrahabr.ru/post/105220/
4) Воронцов SMV: http://www.ccas.ru/voron/download/SVM.pdf
